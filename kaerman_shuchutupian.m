clc; clear; close all;
warning('off', 'MATLAB:handle_graphics:exceptions:SceneNode');

%% =============== 在此处设置您的文件路径 ===============
% 请修改以下路径为您的实际文件路径

% 原始回波数据文件路径（请确保是DAT文件）
rawDataFile = 'E:\1_Label_1.dat';

% 点迹文件路径（请确保是TXT文件）
pointFile = 'E:\PointTracks_1_1_21.txt';

% 航迹文件路径（请确保是TXT文件）
trackFile = 'E:\Tracks_1_1_21.txt';

% =============== 在此处设置图片输出目录 ===============
% 处理后的图片将保存到此文件夹
outputDir = 'E:\picture';

%% 全局变量声明
global stop_flag;
stop_flag = -1;
Fs = 20e6;              % 采样率 (20 MHz)
delta_R = 3e8/2/Fs;     % 距离分辨率

%% 结果保存设置
SAVE_IMAGES = true;     % 保存处理后的图像

% 确保输出目录存在
if SAVE_IMAGES
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
        fprintf('已创建输出目录: %s\n', outputDir);
    end
end

%% 读取数据
[ColPoint, ColTrack] = funcColIndex();
[fid_rawData, pointData, trackData] = funcReadData(ColPoint, ColTrack, rawDataFile, pointFile, trackFile);

%% 初始化卡尔曼滤波器
initialState = [trackData.R(1); trackData.AZ(1); 0; 0]; % [距离; 方位; 距离速度; 方位速度]
kalmanFilter = initKalmanFilter(initialState);

%% 主处理循环
lastPointIndex = 0;
frameCount = 0;

% 创建隐藏figure用于保存图像
if SAVE_IMAGES
    hFig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [0, 0, 800, 600]);
    hAx = axes('Parent', hFig);
end

try
    while ~feof(fid_rawData)
        if stop_flag == 0
            break;
        end
        
        [para, data_out] = funcRawDataParser(fid_rawData);
        
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
        rangeBinIndex = para.Track_No_info(3);
        
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
        
        % ==== 保存处理结果为图片 ====
        if SAVE_IMAGES && ~isempty(Range_plot) && ~isempty(Vr)
            try
                cla(hAx);
                imagesc(hAx, Vr, Range_plot, db(MTD_after));
                axis(hAx, 'xy');
                title(hAx, sprintf('杂波抑制后RD图 (帧号: %d)', frameCount));
                xlabel(hAx, '多普勒速度(米/秒)');
                ylabel(hAx, '距离(米)');
                colorbar('peer', hAx);
                set(hAx, 'XLim', [-30, 30]);
                set(hAx, 'YLim', [Range_plot(1), Range_plot(end)]);
                
                saveas(hFig, fullfile(outputDir, sprintf('%d.png', frameCount)));
                fprintf('已保存帧图像: %d.png\n', frameCount);
            catch
                warning('保存帧图像时出错');
            end
        end
    end
catch ME
    disp(['程序出错: ' ME.message]);
    for k = 1:length(ME.stack)
        disp(['  位置: ' ME.stack(k).name ', 行号: ' num2str(ME.stack(k).line)]);
    end
end

% 清理资源
if SAVE_IMAGES
    close(hFig);
    fprintf('处理完成! 所有图像已保存至: %s\n', outputDir);
end
disp("数据处理完成。");
if ~isempty(fid_rawData) && fid_rawData > 0
    fclose(fid_rawData);
end
clear global stop_flag;

%% % 其他功能函数 % %%
function [ColPoint, ColTrack] = funcColIndex()
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

function [fid_rawData, pointData, trackData] = funcReadData(ColPoint, ColTrack, rawDataFile, pointFile, trackFile)
    % 打开原始回波数据文件
    fid_rawData = fopen(rawDataFile, 'r');
    if fid_rawData == -1
        error('无法打开原始回波文件: %s', rawDataFile);
    end
    fprintf('已打开原始回波文件: %s\n', rawDataFile);
    
    % 读取点迹数据
    if exist(pointFile, 'file')
        pointData = readtable(pointFile, "ReadVariableNames", false);
        pointData.Properties.VariableNames = fieldnames(ColPoint);
        fprintf('已读取点迹文件: %s (%d 个点迹)\n', pointFile, height(pointData));
    else
        error('点迹文件不存在: %s', pointFile);
    end
    
    % 读取航迹数据
    if exist(trackFile, 'file')
        trackData = readtable(trackFile, "ReadVariableNames", false);
        trackData.Properties.VariableNames = fieldnames(ColTrack);
        fprintf('已读取航迹文件: %s (%d 个航迹点)\n', trackFile, height(trackData));
    else
        error('航迹文件不存在: %s', trackFile);
    end
end

function [para, data_out] = funcRawDataParser(fid)
    para = []; 
    data_out = [];
    frame_head = hex2dec('FA55FA55');
    frame_end = hex2dec('55FA55FA');
    
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
    pointNum_in_bowei = data_temp1(3);
    
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
    para.data_length = data_temp(pointNum_in_bowei*4+5);
    
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
    kalmanFilter.State = kalmanFilter.StateTransitionModel * kalmanFilter.State;
    kalmanFilter.StateCovariance = kalmanFilter.StateTransitionModel * ...
                                  kalmanFilter.StateCovariance * ...
                                  kalmanFilter.StateTransitionModel' + ...
                                  kalmanFilter.ProcessNoise;
end

function kalmanFilter = kalmanCorrect(kalmanFilter, measurement)
    % 卡尔曼滤波校正步骤
    S = kalmanFilter.MeasurementModel * kalmanFilter.StateCovariance * kalmanFilter.MeasurementModel' + ...
        kalmanFilter.MeasurementNoise;
    K = kalmanFilter.StateCovariance * kalmanFilter.MeasurementModel' / S;
    innovation = measurement - kalmanFilter.MeasurementModel * kalmanFilter.State;
    kalmanFilter.State = kalmanFilter.State + K * innovation;
    I = eye(size(kalmanFilter.StateCovariance));
    kalmanFilter.StateCovariance = (I - K * kalmanFilter.MeasurementModel) * kalmanFilter.StateCovariance;
end