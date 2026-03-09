clc; clear; close all;
warning('off', 'MATLAB:handle_graphics:exceptions:SceneNode');

%% 全局变量声明
global stop_flag;
stop_flag = -1;
Fs = 20e6;              % 采样率 (20 MHz)
delta_R = 3e8/2/Fs;     % 距离分辨率

%% 结果保存设置（可选）
% 设置为 true 以保存处理结果，设置为 false 则不保存
SAVE_RESULTS = true; % 修改此变量控制是否保存结果

%% 读取数据
[ColPoint,ColTrack] = funcColIndex();
[fid_rawData,pointData,trackData] = funcReadData(ColPoint,ColTrack);

%% 创建图窗 - 只显示抑制后的RD图
[mainFig, trackAx, RDMapAx, trackPlot, pointPlot, mtdImage, targetPlot, kalmanRDPlot] = funcCreateSingleFigure(trackData);

%% 初始化卡尔曼滤波器
initialState = [trackData.R(1); trackData.AZ(1); 0; 0]; % [距离; 方位; 距离速度; 方位速度]
kalmanFilter = initKalmanFilter(initialState);

%% 主处理循环
lastPointIndex = 0;
frameCount = 0;

% 初始化结果存储结构
if SAVE_RESULTS
    processedResults = struct(...
        'frameCount', {}, ...
        'MTD_after', {}, ...
        'Vr', {}, ...
        'Range_plot', {}, ...
        'predictedPosition', {}, ...
        'detectedPosition', {} ...
    );
end

try
    while ~feof(fid_rawData) && ishghandle(mainFig)
        if stop_flag == 0
            break;
        end
        
        [para,data_out] = funcRawDataParser(fid_rawData);
        
        if isempty(para) || isempty(data_out)
            continue;
        end
        
        frameCount = frameCount + 1;
        fprintf('处理帧号: %d\n', frameCount);
        
        % ==== 获取当前航迹点序号 ====
        index_trackPointNo = min(para.Track_No_info(2), height(trackData));
        
        % ==== 卡尔曼滤波处理 ====
        % 预测
        kalmanFilter = kalmanPredict(kalmanFilter);
        predictedState = kalmanFilter.State;
        
        % 校正 (仅当有新点时)
        if lastPointIndex ~= index_trackPointNo
            measurement = [trackData.R(index_trackPointNo); trackData.AZ(index_trackPointNo)];
            kalmanFilter = kalmanCorrect(kalmanFilter, measurement);
            lastPointIndex = index_trackPointNo;
        end
        
        % ==== MTD处理 (抑制前) ====
        % 准备加窗
        MTD_win = taylorwin(size(data_out, 2), [], -30);
        coef_MTD_2D = repmat(MTD_win, [1, size(data_out, 1)]);
        coef_MTD_2D = permute(coef_MTD_2D, [2, 1]);
        
        % 应用窗函数并FFT
        data_proc_MTD_win_out = data_out .* coef_MTD_2D;
        MTD_before = fftshift(fft(data_proc_MTD_win_out, [], 2), 2);
        
        % ==== 自适应杂波抑制 ====
        delta_Vr = 3e8 / (2 * size(data_out, 2) * para.PRT * para.Freq);
        Vr = (-size(data_out, 2)/2 : size(data_out, 2)/2 - 1) * delta_Vr;
        
        % 预测目标在RD图上的位置
        predictedRange = predictedState(1);
        rangeBinIndex = para.Track_No_info(3); % 使用原始距离单元
        
        % 使用点迹数据的真实多普勒速度
        predictedDoppler = pointData.Doppler(index_trackPointNo);
        [~, dopplerBinIndex] = min(abs(Vr - predictedDoppler));
        
        % 创建自适应抑制窗
        suppressionRadius = 5;
        suppressionWindow = ones(size(MTD_before));
        
        % 计算实际距离范围
        range_start_bin = rangeBinIndex - 15;
        range_end_bin = rangeBinIndex + 15;
        Range_plot = (range_start_bin:range_end_bin) * delta_R;
        
        % 应用锥形抑制窗
        local_rangeBinIndex = rangeBinIndex - range_start_bin + 1;
        local_dopplerBinIndex = dopplerBinIndex;
        
        for r = max(1, local_rangeBinIndex-suppressionRadius):min(size(suppressionWindow,1), local_rangeBinIndex+suppressionRadius)
            for d = max(1, local_dopplerBinIndex-suppressionRadius):min(size(suppressionWindow,2), local_dopplerBinIndex+suppressionRadius)
                dist = sqrt((r - local_rangeBinIndex)^2 + (d - local_dopplerBinIndex)^2);
                if dist <= suppressionRadius
                    suppressionWindow(r, d) = 0.1 + 0.9 * (dist / suppressionRadius);
                end
            end
        end
        
        % 应用抑制窗 (创建抑制后的数据)
        MTD_after = MTD_before .* suppressionWindow;
        
        % ==== 目标检测 ====
        Amp_max_Vr_unit = para.Track_No_info(4);
        Amp_max_Vr_unit = (Amp_max_Vr_unit > size(MTD_after, 2)/2) .* ...
            (Amp_max_Vr_unit - size(MTD_after, 2)/2) + ...
            (Amp_max_Vr_unit <= size(MTD_after, 2)/2) .* ...
            (Amp_max_Vr_unit + size(MTD_after, 2)/2);
        
        center_local_bin = 16;
        local_radius = 5;
        
        % 计算局部检测窗口
        range_start_local = max(1, center_local_bin - local_radius);
        range_end_local = min(size(MTD_after, 1), center_local_bin + local_radius);
        doppler_start = max(1, Amp_max_Vr_unit - local_radius);
        doppler_end = min(size(MTD_after, 2), Amp_max_Vr_unit + local_radius);
        
        Target_sig = MTD_after(range_start_local:range_end_local, doppler_start:doppler_end);
        
        % 检测峰值
        [Amp_max_index_row, Amp_max_index_col] = find(abs(Target_sig) == max(max(abs(Target_sig))), 1);
        
        % 转换到全局距离位置
        detected_range_bin = range_start_local + Amp_max_index_row - 1;
        Amp_max_range = Range_plot(detected_range_bin);
        Amp_max_Vr = Vr(doppler_start + Amp_max_index_col - 1);
        
        % ==== 存储处理结果 ====
        if SAVE_RESULTS
            processedResults(frameCount).frameCount = frameCount;
            processedResults(frameCount).MTD_after = MTD_after;
            processedResults(frameCount).Vr = Vr;
            processedResults(frameCount).Range_plot = Range_plot;
            processedResults(frameCount).predictedPosition = [local_rangeBinIndex, local_dopplerBinIndex];
            processedResults(frameCount).detectedPosition = [detected_range_bin, Amp_max_Vr_unit];
        end
        
        % ==== 图形更新 ====
        % 1. 更新航迹图
        if ishghandle(pointPlot)
            set(pointPlot, 'XData', trackData.AZ(index_trackPointNo), ...
                'YData', trackData.R(index_trackPointNo));
        end
        
        % 2. 更新卡尔曼滤波预测点（航迹图）
        if ishghandle(trackAx)
            kalmanTrackPlot = plot(trackAx, predictedState(2), predictedState(1), 'gx', ...
                'MarkerSize', 10, 'LineWidth', 1.5);
        end

        % 3. 更新MTD图像 - 抑制后
        if ishghandle(mtdImage)
            set(mtdImage, 'CData', db(MTD_after), ...
                'XData', Vr, ...
                'YData', Range_plot);
        end
        
        % 4. 更新目标标记
        if ishghandle(targetPlot)
            set(targetPlot, 'XData', Amp_max_Vr, 'YData', Amp_max_range);
        end
        
        % 5. 更新卡尔曼预测标记
        if ishghandle(kalmanRDPlot)
            set(kalmanRDPlot, 'XData', Vr(local_dopplerBinIndex), ...
                'YData', Range_plot(local_rangeBinIndex));
        end
        
        % 6. 设置坐标轴范围
        if ishghandle(RDMapAx)
            set(RDMapAx, 'XLim', [-30, 30], 'YLim', [Range_plot(1), Range_plot(end)]);
            title(RDMapAx, sprintf('杂波抑制后RD图 (帧号: %d)', frameCount), ...
                'FontSize', 12, 'FontWeight', 'bold');
        end

        % 7. 添加详细信息标注
        if ishghandle(mainFig)
            infoStr = {
                sprintf('距离: %.1fm(测量)/%.1fm(预测)', ...
                    trackData.R(index_trackPointNo), predictedRange);
                sprintf('速度: %.2fm/s(测量)/%.2fm/s(预测)', ...
                    pointData.Doppler(index_trackPointNo), predictedDoppler);
            };
            
            % 查找或创建信息框
            infoBox = findobj(mainFig, 'Tag', 'InfoBox');
            if isempty(infoBox) || ~ishghandle(infoBox)
                annotation('textbox', [0.38, 0.75, 0.22, 0.07], ...
                    'String', infoStr, 'Tag', 'InfoBox', ...
                    'BackgroundColor', 'white', 'EdgeColor', 'black', ...
                    'FontSize', 10);
            else
                set(infoBox, 'String', infoStr);
            end
        end
        
        drawnow;
        pause(0.1);

        % 删除临时绘图对象
        if exist('kalmanTrackPlot', 'var') && ishghandle(kalmanTrackPlot)
            delete(kalmanTrackPlot);
        end
    end
    
    % 保存处理结果（可选）
    if SAVE_RESULTS
        [file, path] = uiputfile('processedResults.mat', '保存处理结果');
        if ischar(file)
            save(fullfile(path, file), 'processedResults', '-v7.3');
            fprintf('处理结果已保存至: %s\n', fullfile(path, file));
        end
    end
catch ME
    disp(['程序出错: ' ME.message]);
    for k = 1:length(ME.stack)
        disp(['  位置: ' ME.stack(k).name ', 行号: ' num2str(ME.stack(k).line)]);
    end
end

disp("已完成解析。");
if ~isempty(fid_rawData) && fid_rawData > 0
    fclose(fid_rawData);
end
clear global stop_flag;

%% % 单图显示创建函数 % %%
function [mainFig, trackAx, RDMapAx, trackPlot, pointPlot, mtdImage, targetPlot, kalmanRDPlot] = funcCreateSingleFigure(trackData)
    % 创建图形界面
    mainFig = figure("Name", '雷达杂波抑制结果', ...
        'NumberTitle', 'off', 'Position', [50, 50, 1200, 600], ...
        'Color', [0.95 0.95 0.95]);
    movegui(mainFig, 'center');
    
    % 按钮
    uicontrol('String', '暂停', 'Position', [20, 5, 80, 30], 'Callback', 'uiwait(gcf)', ...
        'BackgroundColor', [0.9 0.9 1], 'FontWeight', 'bold');
    uicontrol('String', '继续', 'Position', [120, 5, 80, 30], 'Callback', 'uiresume(gcf)', ...
        'BackgroundColor', [0.9 1 0.9], 'FontWeight', 'bold');
    uicontrol('String', '停止', 'Position', [220, 5, 80, 30], 'Callback', @buttonStop, ...
        'BackgroundColor', [1 0.9 0.9], 'FontWeight', 'bold');
    
    % 航迹显示区域 (左侧)
    trackAx = subplot(1, 2, 1);
    trackPlot = plot(trackAx, trackData.AZ, trackData.R, 'b-', 'LineWidth', 1.5);
    grid(trackAx, "on");
    hold(trackAx, "on");
    
    % 添加标题和标签
    title(trackAx, "目标航迹", 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(trackAx, "方位(度)", 'FontSize', 10);
    ylabel(trackAx, "距离(米)", 'FontSize', 10);
    
    % 当前点 (红色圆圈)
    pointPlot = plot(trackAx, NaN, NaN, 'ro', ...
        'MarkerSize', 8, 'LineWidth', 1.5, ...
        'DisplayName', '当前点');
    
    % 添加图例
    legend(trackAx, 'Location', 'best');
    
    % RD图显示区域 (右侧)
    RDMapAx = subplot(1, 2, 2);
    
    % 创建伪图像用于初始化
    mtdImage = imagesc(RDMapAx, 0, 0, zeros(2,2));
    colorbar(RDMapAx);
    hold(RDMapAx, "on");
    
    % 目标检测标记 (红色五角星)
    targetPlot = plot(RDMapAx, NaN, NaN, 'r*', ...
        'MarkerSize', 10, 'LineWidth', 1.5, ...
        'DisplayName', '检测目标');
    
    % 卡尔曼预测标记 (蓝色圆圈)
    kalmanRDPlot = plot(RDMapAx, NaN, NaN, 'bo', ...
        'MarkerSize', 10, 'LineWidth', 1.5, ...
        'DisplayName', '卡尔曼预测');
    
    % 添加标题和标签
    title(RDMapAx, "杂波抑制后RD图", 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(RDMapAx, "多普勒速度(米/秒)", 'FontSize', 10);
    ylabel(RDMapAx, "距离(米)", 'FontSize', 10);
    
    % 添加图例
    legend(RDMapAx, 'Location', 'best');
    
    % 设置坐标轴方向
    set(RDMapAx, 'YDir', 'normal');
    
    % 添加可视化说明
    annotation('textbox', [0.05, 0.95, 0.9, 0.04], ...
        'String', '图例说明: 蓝色圆圈 = 卡尔曼预测位置 | 红色五角星 = 实际检测点', ...
        'EdgeColor', 'none', 'FontSize', 11, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [1 1 0.8]);
end

%% % 其他功能函数 % %%
function [ColPoint,ColTrack] = funcColIndex()
    % 点迹数据列索引
    ColPoint.Time = 1;      % 点时间
    ColPoint.TrackID = 2;   % 航迹批号
    ColPoint.R = 3;         % 距离
    ColPoint.AZ = 4;        % 方位
    ColPoint.EL = 5;        % 俯仰
    ColPoint.Doppler = 6;   % 多普勒速度
    ColPoint.Amp = 7;       % 和幅度
    ColPoint.SNR = 8;       % 信噪比
    ColPoint.PointNum = 9;  % 原始点数量
    
    % 航迹数据列索引
    ColTrack.Time = 1;      % 点时间
    ColTrack.TrackID = 2;   % 航迹批号
    ColTrack.R = 3;         % 滤波距离
    ColTrack.AZ = 4;        % 滤波方位
    ColTrack.EL = 5;        % 滤波俯仰
    ColTrack.Speed = 6;     % 全速度
    ColTrack.Vx = 7;        % X向速度(东)
    ColTrack.Vy = 8;        % Y向速度(北)
    ColTrack.Vz = 9;        % Z向速度(天)
    ColTrack.Head = 10;     % 航向角
end

function [fid_rawData,pointData,trackData] = funcReadData(ColPoint,ColTrack)
    % 设置文件路径
    selectedDir = uigetdir(pwd, '选择数据根目录');
    if selectedDir == 0
        error('错误！未选择文件路径。');
    end
    
    IQDataDir = fullfile(selectedDir, '原始回波');
    TrackDir = fullfile(selectedDir, '航迹');
    PointDir = fullfile(selectedDir, '点迹');
    
    if ~all([exist(IQDataDir,"dir"),exist(TrackDir,"dir"),exist(PointDir,"dir")])
        error('错误！路径下缺少必要的文件夹。');
    end
    
    % 原始回波文件
    [fileName,filePath] = uigetfile('*_Label_*.dat','选择数据',IQDataDir);
    if fileName == 0
        error('错误！未选择原始回波文件。');
    end
    rawDataFile = fullfile(filePath,fileName);
    fid_rawData = fopen(rawDataFile,'r');
    if fid_rawData == -1
        error(['无法打开文件: ' rawDataFile]);
    end
    
    % 数据批号与标签值
    tokens = regexp(fileName, '^(\d+)_Label_(\d+)\.dat$', 'tokens');
    if isempty(tokens)
        error('文件名格式错误: %s', fileName);
    end
    tokens = tokens{1};
    track_No = str2double(tokens{1});
    label = str2double(tokens{2});
    
    % 点迹文件
    pointPattern = sprintf('PointTracks_%d_%d_*.txt', track_No, label);
    pointFile = dir(fullfile(PointDir, pointPattern));
    if isempty(pointFile)
        error('点迹文件不存在: %s', fullfile(PointDir, pointPattern));
    end
    pointFile = fullfile(pointFile(1).folder, pointFile(1).name);
    pointData = readtable(pointFile, "ReadVariableNames", false);
    pointData.Properties.VariableNames = fieldnames(ColPoint);
    
    % 航迹文件
    trackPattern = sprintf('Tracks_%d_%d_*.txt', track_No, label);
    trackFile = dir(fullfile(TrackDir, trackPattern));
    if isempty(trackFile)
        error('航迹文件不存在: %s', fullfile(TrackDir, trackPattern));
    end
    trackFile = fullfile(trackFile(1).folder, trackFile(1).name);
    trackData = readtable(trackFile, "ReadVariableNames", false);
    trackData.Properties.VariableNames = fieldnames(ColTrack);
    
    disp("已读取输入数据。");
end

function [para, data_out] = funcRawDataParser(fid)
    % 读取解析原始回波数据
    para = [];
    data_out = [];
    
    frame_head = hex2dec('FA55FA55');
    frame_end = hex2dec('55FA55FA');
    
    % 查找帧头
    head_find = fread(fid, 1, 'uint32');
    if isempty(head_find) || feof(fid)
        return;
    end
    
    while head_find ~= frame_head && ~feof(fid)
        fseek(fid, -3, 'cof');
        head_find = fread(fid, 1, 'uint32');
        if feof(fid)
            return;
        end
    end
    
    frame_data_length = fread(fid, 1, 'uint32');
    if isempty(frame_data_length)
        return;
    end
    frame_data_length = frame_data_length * 4;
    
    % 检查帧尾
    fseek(fid, frame_data_length - 12, 'cof');
    end_find = fread(fid, 1, 'uint32');
    if isempty(end_find) || end_find ~= frame_end
        return;
    end
    
    fseek(fid, -frame_data_length + 4, 'cof');
    
    % 读取参数
    data_temp1 = fread(fid, 3, 'uint32');
    if length(data_temp1) < 3
        return;
    end
    para.E_scan_Az = data_temp1(2) * 0.01;
    pointNum_in_bowei = data_temp1(3);  % 修复变量名错误
    
    % 读取轨道信息
    data_temp = fread(fid, pointNum_in_bowei * 4 + 5, 'uint32');
    if length(data_temp) < pointNum_in_bowei * 4 + 5
        return;
    end
    para.Track_No_info = data_temp(1:pointNum_in_bowei*4);
    para.Freq = data_temp(pointNum_in_bowei*4+1) * 1e6;
    para.CPIcount = data_temp(pointNum_in_bowei*4+2);
    para.PRTnum = data_temp(pointNum_in_bowei*4+3);
    para.PRT = data_temp(pointNum_in_bowei*4+4) * 0.0125e-6;
    para.data_length = data_temp(pointNum_in_bowei*4+5);  % 修复变量名错误
    
    % 读取数据
    data_out_temp = fread(fid, para.PRTnum * 31 * 2, 'float');
    if isempty(data_out_temp) || feof(fid)
        return;
    end
    
    % 处理数据
    data_out_real = data_out_temp(1:2:end);
    data_out_imag = data_out_temp(2:2:end);
    data_out_complex = data_out_real + 1i * data_out_imag;
    data_out = reshape(data_out_complex, 31, para.PRTnum);  % 距离×PRT数
    
    % 跳过帧尾
    fseek(fid, 4, 'cof');
end

function kalmanFilter = initKalmanFilter(initialState)
    % 卡尔曼滤波器初始化
    dt = 1; % 时间步长(假设1秒)
    
    % 状态转移矩阵 (匀速运动模型)
    F = [1 0 dt 0;
         0 1 0 dt;
         0 0 1 0;
         0 0 0 1];
    
    % 测量矩阵 (只能观测位置)
    H = [1 0 0 0;
         0 1 0 0];
    
    % 过程噪声协方差矩阵
    Q = diag([0.1, 0.1, 0.5, 0.5]);
    
    % 测量噪声协方差矩阵
    R = diag([10, 10]); % 距离和方位的测量噪声
    
    % 初始状态协方差
    P = eye(4);
    
    % 创建卡尔曼滤波器结构体
    kalmanFilter = struct();
    kalmanFilter.State = initialState;
    kalmanFilter.StateCovariance = P;
    kalmanFilter.StateTransitionModel = F;
    kalmanFilter.MeasurementModel = H;
    kalmanFilter.ProcessNoise = Q;
    kalmanFilter.MeasurementNoise = R;
    
    disp("卡尔曼滤波器已初始化.");
end

function kalmanFilter = kalmanPredict(kalmanFilter)
    % 卡尔曼滤波预测步骤
    % 状态预测
    kalmanFilter.State = kalmanFilter.StateTransitionModel * kalmanFilter.State;
    
    % 协方差预测
    kalmanFilter.StateCovariance = kalmanFilter.StateTransitionModel * ...
                                  kalmanFilter.StateCovariance * ...
                                  kalmanFilter.StateTransitionModel' + ...
                                  kalmanFilter.ProcessNoise;
end

function kalmanFilter = kalmanCorrect(kalmanFilter, measurement)
    % 卡尔曼滤波校正步骤
    % 计算卡尔曼增益
    S = kalmanFilter.MeasurementModel * kalmanFilter.StateCovariance * kalmanFilter.MeasurementModel' + ...
        kalmanFilter.MeasurementNoise;
    K = kalmanFilter.StateCovariance * kalmanFilter.MeasurementModel' / S;
    
    % 状态更新
    innovation = measurement - kalmanFilter.MeasurementModel * kalmanFilter.State;
    kalmanFilter.State = kalmanFilter.State + K * innovation;
    
    % 协方差更新
    I = eye(size(kalmanFilter.StateCovariance));
    kalmanFilter.StateCovariance = (I - K * kalmanFilter.MeasurementModel) * kalmanFilter.StateCovariance;
end

function buttonStop(~, ~)
    % 停止按钮回调函数
    global stop_flag;
    stop_flag = 0;
    disp("处理已停止.");
end