clc; clear; close all;
warning('off', 'MATLAB:handle_graphics:exceptions:SceneNode');

%% =============== 在此设置文件夹路径 ===============
% 第一个文件夹（包含点迹、航迹、原始回波的文件夹）
folderA = "D:\2";

% 第二个文件夹（包含1-6子文件夹的文件夹）
folderB = "D:\3";

%% 全局变量声明
global stop_flag;
stop_flag = -1;
Fs = 20e6;              % 采样率 (20 MHz)
delta_R = 3e8/2/Fs;     % 距离分辨率

%% 获取原始回波文件列表
rawDataPath = fullfile(folderA, '原始回波');
rawFiles = dir(fullfile(rawDataPath, '*.dat'));

% 检查是否有文件
if isempty(rawFiles)
    error('在原始回波文件夹中未找到任何dat文件: %s', rawDataPath);
end

%% 逐个处理每个回波文件
for fileIdx = 1:length(rawFiles)
    try
        rawDataFile = fullfile(rawFiles(fileIdx).folder, rawFiles(fileIdx).name);
        
        % 从文件名中提取序号1和序号2
        [~, fileName, ~] = fileparts(rawDataFile);
        tokens = regexp(fileName, '^(\d+)_Label_(\d+)$', 'tokens');
        
        if isempty(tokens)
            warning('文件名格式错误: %s。跳过此文件。', fileName);
            continue;
        end
        
        tokens = tokens{1};
        serial1 = tokens{1};
        serial2 = tokens{2};
        
        fprintf('\n处理文件: %s\n', fileName);
        fprintf('序号1: %s, 序号2: %s\n', serial1, serial2);
        
        % 创建本文件的输出文件夹 (folderB中的serial2子文件夹)
        outputDir = fullfile(folderB, serial2);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
            fprintf('已创建输出目录: %s\n', outputDir);
        end
        
        % 查找对应的点迹和航迹文件
        pointPattern = sprintf('PointTracks_%s_%s_*.txt', serial1, serial2);
        trackPattern = sprintf('Tracks_%s_%s_*.txt', serial1, serial2);
        
        pointFiles = dir(fullfile(folderA, '点迹', pointPattern));
        trackFiles = dir(fullfile(folderA, '航迹', trackPattern));
        
        if isempty(pointFiles)
            warning('未找到点迹文件: %s。跳过此文件。', pointPattern);
            continue;
        end
        
        if isempty(trackFiles)
            warning('未找到航迹文件: %s。跳过此文件。', trackPattern);
            continue;
        end
        
        % 使用找到的第一个点迹和航迹文件
        pointFile = fullfile(pointFiles(1).folder, pointFiles(1).name);
        trackFile = fullfile(trackFiles(1).folder, trackFiles(1).name);
        
        fprintf('点迹文件: %s\n', pointFile);
        fprintf('航迹文件: %s\n', trackFile);
        
        %% 读取数据
        [ColPoint, ColTrack] = funcColIndex();
        [fid_rawData, pointData, trackData] = funcReadData(ColPoint, ColTrack, rawDataFile, pointFile, trackFile);
        
        %% 主处理循环
        frameCount = 0;
        
        % 创建隐藏figure用于保存图像
        hFig = figure('Visible', 'off', 'Units', 'pixels', 'Position', [0, 0, 800, 600]);
        hAx = axes('Parent', hFig);
        
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
                
                % ==== 获取当前航迹点序号 ====
                index_trackPointNo = min(para.Track_No_info(2), height(trackData));
                
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
                
                % 使用点迹数据的真实多普勒速度
                trueDoppler = pointData.Doppler(index_trackPointNo);
                [~, dopplerBinIndex] = min(abs(Vr - trueDoppler));
                
                % 计算实际距离范围
                rangeBinIndex = para.Track_No_info(3);
                range_start_bin = rangeBinIndex - 15;
                range_end_bin = rangeBinIndex + 15;
                Range_plot = (range_start_bin:range_end_bin) * delta_R;
                
                % 创建自适应抑制窗
                suppressionRadius = 5;
                suppressionWindow = ones(size(MTD_before));
                
                % 应用锥形抑制窗
                for r = max(1, rangeBinIndex-suppressionRadius):min(size(suppressionWindow,1), rangeBinIndex+suppressionRadius)
                    for d = max(1, dopplerBinIndex-suppressionRadius):min(size(suppressionWindow,2), dopplerBinIndex+suppressionRadius)
                        dist = sqrt((r - rangeBinIndex)^2 + (d - dopplerBinIndex)^2);
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
                if ~isempty(Range_plot) && ~isempty(Vr)
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
                        
                        imgName = sprintf('%s_%s_%d.png', serial1, serial2, frameCount);
                        imgPath = fullfile(outputDir, imgName);
                        saveas(hFig, imgPath);
                        fprintf('已保存帧图像: %s\n', imgName);
                    catch
                        warning('保存帧图像时出错');
                    end
                end
            end
        catch ME
            disp(['处理文件时出错: ' ME.message]);
            for k = 1:length(ME.stack)
                disp(['  位置: ' ME.stack(k).name ', 行号: ' num2str(ME.stack(k).line)]);
            end
        end
        
        % 清理当前文件资源
        close(hFig);
        if ~isempty(fid_rawData) && fid_rawData > 0
            fclose(fid_rawData);
        end
        fprintf('已完成文件处理: %s\n\n', fileName);
        
    catch ME
        disp(['处理过程中出错: ' ME.message]);
    end
end

disp("所有文件处理完成!");
clear global stop_flag;

%% % 其他功能函数 % %%
function [ColPoint, ColTrack] = funcColIndex()
    ColPoint.Time = 1;      ColPoint.TrackID = 2;
    ColPoint.R = 3;         ColPoint.AZ = 4;
    ColPoint.EL = 5;        ColPoint.Doppler = 6;
    ColPoint.Amp = 7;       ColPoint.SNR = 8;
    ColPoint.PointNum = 9;
    
    ColTrack.Time = 1;      ColTrack.TrackID = 2;
    ColTrack.R = 3;         ColTrack.AZ = 4;
    ColTrack.EL = 5;        ColTrack.Speed = 6;
    ColTrack.Vx = 7;        ColTrack.Vy = 8;
    ColTrack.Vz = 9;        ColTrack.Head = 10;
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
        fprintf('已读取点迹文件: %s (%d 行)\n', pointFile, height(pointData));
    else
        error('点迹文件不存在: %s', pointFile);
    end
    
    % 读取航迹数据
    if exist(trackFile, 'file')
        trackData = readtable(trackFile, "ReadVariableNames", false);
        trackData.Properties.VariableNames = fieldnames(ColTrack);
        fprintf('已读取航迹文件: %s (%d 行)\n', trackFile, height(trackData));
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
    data_out = reshape(data_out_complex, 31, para.PRTnum);
    
    % 跳过帧尾
    fseek(fid, 4, 'cof');
end