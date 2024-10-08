%% Drop frames
% change parameters!!!
x_off = find(ops.xoff>20 | ops.xoff<-20);  %35/20
y_off = find(ops.yoff>20 | ops.yoff<-20);
xy_off = sort([x_off,y_off]);
xy_uniq = unique(xy_off);

% replace uncontinuous dropped frames with regression value
noncont_all = [];
for i = 1:length(xy_uniq)-2
    if xy_uniq(i+1)-xy_uniq(i)~=1 && xy_uniq(i+1)-xy_uniq(i+2)~=-1
        noncont = xy_uniq(i+1);
        noncont_all(end+1) = noncont;
    end
end
noncont_all = [xy_uniq(1),noncont_all];
for j = 2:length(noncont_all)-1
    for k = 1:size(F,1)
        F(k,noncont_all(j)) = mean([F(k,noncont_all(j)+1),F(k,noncont_all(j)-1)]);
        Fneu(k,noncont_all(j)) = mean([Fneu(k,noncont_all(j)+1),Fneu(k,noncont_all(j)-1)]);
        ops.xoff(1,noncont_all(j)) = mean([ops.xoff(1,noncont_all(j)+1),ops.xoff(1,noncont_all(j)-1)]);
        ops.yoff(1,noncont_all(j)) = mean([ops.yoff(1,noncont_all(j)+1),ops.yoff(1,noncont_all(j)-1)]);
    end
end

% replace 2 continuous dropped frames with regression value
cont2_all = [];
diff_uniq = diff(xy_uniq)==1;
for m = 1:length(diff_uniq)-2
    if diff_uniq(m+1)-diff_uniq(m)==1 && diff_uniq(m+1)-diff_uniq(m+2)==1
        cont2 = xy_uniq(m+1);
        cont2_all(end+1) = cont2;
    end
end
for n = 1:length(cont2_all)
    for k = 1:size(F,1)
        F(k,cont2_all(n)) = mean([F(k,cont2_all(n)-1),F(k,cont2_all(n)+2)]);
        F(k,cont2_all(n)+1) = mean([F(k,cont2_all(n)-1),F(k,cont2_all(n)+2)]);
        Fneu(k,cont2_all(n)) = mean([Fneu(k,cont2_all(n)-1),Fneu(k,cont2_all(n)+2)]);
        Fneu(k,cont2_all(n)+1) = mean([Fneu(k,cont2_all(n)-1),Fneu(k,cont2_all(n)+2)]);
    end
end
cont2_num = sort([cont2_all,cont2_all+1]);

cont3 = sort([setdiff(xy_uniq,[noncont_all,cont2_num])]);
zero_vec = zeros(1,size(F,2));
zero_vec(cont3)=1;
zero_mat = reshape(zero_vec,[size(F,2)/32,32]);
% zero_mat = reshape(zero_vec,[size(F,2)/24,24]);


col_all = {};
for p = 1:size(zero_mat,2)
    col = find(zero_mat(:,p) == 1);
    col_all{end+1} = col;
end

%% find iscells from suite2p; calculate daily baseline

imaging_day = 32; % imaging session

iscell_index = find(iscell(:,1)==1);

base_per_day = {};
for k = 1:length(iscell_index)
    data_per_day1 = reshape(F(iscell_index(k),:),[size(F,2)/imaging_day],imaging_day);        % change the number of 10-minute session
    data_per_day2 = reshape(Fneu(iscell_index(k),:),[size(Fneu,2)/imaging_day],imaging_day);
    data_per_day = data_per_day1 - 0.7*data_per_day2; 
%     data_per_day = reshape(F(iscell_index(k),:),[size(F,2)/32],32);
%     data_per_day = reshape(Fneu(iscell_index(k),:),[size(Fneu,2)/32],32);
    
    frame_num = size(data_per_day,1);

    base_per_day{end+1} = data_per_day;
end

% cell shift

cellshift_x = reshape(ops.xoff,[size(F,2)/imaging_day],imaging_day); % change the number of imaging days
cellshift_y = reshape(ops.yoff,[size(F,2)/imaging_day],imaging_day);
cellshift_vector = sqrt(cellshift_x.^2 + cellshift_y.^2);
cellshift_vector3 = [];
for q = 1:size(cellshift_vector,2)
    new_vector = zeros(size(cellshift_vector,1),1);
    for i = 1:size(cellshift_vector,1)-1
        new_vector(i) = cellshift_vector(i+1,q) - cellshift_vector(i,q);
    end
    new_vector = abs(new_vector);
    cellshift_vector3 = [cellshift_vector3,new_vector];
end

% cellshift_vector_z3 = zscore(cellshift_vector);
cellshift_vector2 = {cellshift_vector3};
cellshift_vector_z = repelem(cellshift_vector2,length(iscell_index));

%% align arduino time point with matrix
% Note that different files use either 'Puff/Blank' or 'real/fake'
[~,sheet_name1]=xlsfinfo('Arduino.xlsx');
for k=1:numel(sheet_name1)
  [arduino_parameter{k},arduino{k}]=xlsread('Arduino.xlsx',sheet_name1{k});
end

[~,sheet_name2]=xlsfinfo('Arduino time point.xlsx')
for l=1:numel(sheet_name2)
  arduino_time{l}=xlsread('Arduino time point.xlsx',sheet_name2{l});
end


range_puff_all_page = {};
range_blank_all_page = {};
puff_merge = {};
blank_merge = {};
c_all = {};
pixel_puff_all_page = {};
pixel_blank_all_page = {};
puff_pixel_merge = {};
blank_pixel_merge = {};

for z = 1: length(iscell_index)
    for w = 1:k
    

        framerate_row = 1: length(base_per_day{z});
        framerate = framerate_row'*600000/length(base_per_day{z});
        intensity_base = base_per_day{z}(:,w);
        pixel = cellshift_vector_z{z}(:,w);
        trial_time = arduino_time{w}(:,1);

        b=1;
        for i = 1:length(trial_time)
            for j = 1:length(framerate)-1
                a1 = abs(framerate(b)-trial_time(i));
                a2 = abs(framerate(j+1)-trial_time(i));
                if a1 > a2
                    b = j+1;
                end
            end
            c(i) = b;       
        end

        range = [];
        pixel_range = [];
        for xyz = 1:length(c)
            range_pre = [];
            pixel_pre = [];
            celem = c(xyz);
            for n = (celem-51):1:(celem+50)                                
                range_pre = [range_pre, intensity_base(n)];
                pixel_pre = [pixel_pre, pixel(n)];
            end
            range(xyz, :) = [range_pre];
            pixel_range(xyz, :) = [pixel_pre];
        end
        range_new = range';
        pixel_new = pixel_range';

        puff = find(strcmp(arduino{w},'Puff'));
        blank = find(strcmp(arduino{w},'Blank'));

        range_puff = [];
        pixel_puff = [];
        for v = 1:length(puff)
            range_puff = [range_puff, range_new(:,puff(v))];
            pixel_puff = [pixel_puff, pixel_new(:,puff(v))];
        end

        range_blank = [];
        pixel_blank = [];
        for q = 1:length(blank)
            range_blank = [range_blank, range_new(:,blank(q))];
            pixel_blank = [pixel_blank, pixel_new(:,blank(q))];
        end
        range_puff_all_page{end+1} = range_puff;
        pixel_puff_all_page{end+1} = pixel_puff;
        range_blank_all_page{end+1} = range_blank; 
        pixel_blank_all_page{end+1} = pixel_blank;
        
        c_all{end+1} = c;
        
    end
     
end

all_frame = {};
for q = 1:length(c_all)/length(iscell_index)
    trial_frame = c_all{q};
    all_frame{end+1} = trial_frame;
end

for g = 1:length(range_puff_all_page)/2
            
    puff_page = [range_puff_all_page{g*2-1}';range_puff_all_page{g*2}']; 
    puff_page_pixel = [pixel_puff_all_page{g*2-1}';pixel_puff_all_page{g*2}']; 
            
    puff_merge{end+1} = puff_page';
    puff_pixel_merge{end+1} = puff_page_pixel';
            
end
        
for g = 1:length(range_blank_all_page)/2
            
    blank_page = [range_blank_all_page{g*2-1}';range_blank_all_page{g*2}'];  
    blank_page_pixel = [pixel_blank_all_page{g*2-1}';pixel_blank_all_page{g*2}']; 
            
    blank_merge{end+1} = blank_page';
    blank_pixel_merge{end+1} = blank_page_pixel';
            
end

%calculate the change in pixel shift

d_shift = [];
d_shift_all = {};
for num4 = 1:length(puff_pixel_merge)
    pixel_merge_mat = puff_pixel_merge{num4};
    for num5 = 1:size(pixel_merge_mat,2)
        d_shift_col = abs(diff(pixel_merge_mat(:,num5)));
        d_shift = [d_shift, d_shift_col];    
    end
    d_shift_all{end+1} = d_shift;
    d_shift = [];   
end

d_shift_b = [];
d_shift_all_b = {};
for num6 = 1:length(blank_pixel_merge)
    pixel_merge_mat_b = blank_pixel_merge{num6};
    for num7 = 1:size(pixel_merge_mat_b,2)
        d_shift_col_b = abs(diff(pixel_merge_mat_b(:,num7)));
        d_shift_b = [d_shift_b, d_shift_col_b];    
    end
    d_shift_all_b{end+1} = d_shift_b;
    d_shift_b = [];   
end

%calculate the mean pixel shift for each day
d_shift_all = d_shift_all(1:imaging_day/2);
mean_results = cell(size(d_shift_all));

for i = 1:numel(d_shift_all)
    current_cell = d_shift_all{i};
    
    if isempty(current_cell)
        mean_results{i} = [];
    else
        mean_values = mean(current_cell(52:56, :), 1);
        mean_results{i} = mean_values;
    end
end

averages_vector = zeros(1, numel(mean_results));
for i = 1:numel(mean_results)
    current_cell = mean_results{i};
    average_value = mean(current_cell(:));
    averages_vector(i) = average_value;
end

%% dF/Fo (all stimulus trials)

trial_mat2 = [];
puff_merge2 = {};
pks_trial_all = [];
pks_pos_all = [];
delayResp_all = [];
for s = 1: length(puff_merge)
        for d = 1:size(puff_merge{s},2)
            trial_mat = puff_merge{s}(:,d);
            dF_F = (trial_mat-mean(trial_mat(47:51)))/mean(trial_mat(47:51));
            trial_mat2 = [trial_mat2,dF_F];
            pks_trial = mean(max(trial_mat2(52:56,:))); % mean pks by trial
            [rows, columns] = size(trial_mat2(52:56,:));
            maxRowPositions = zeros(1, columns);
            delay_pks = zeros(1, columns);
            for col = 1:columns
                [~, maxRow] = max(trial_mat2(52:56, col));
                maxRowPositions(col) = (maxRow-1)/5.11; %convert frame number to second
                delay_pks(col) = trial_mat2((52+(maxRow-1)+5), col); %check Ca signal at 1s after the response peak
            end
            pks_pos = mean(maxRowPositions);
            delayResp = mean(delay_pks);
        end
        puff_merge2{end+1} = trial_mat2;
        pks_trial_all(end+1) = pks_trial;
        pks_pos_all(end+1) = pks_pos;
        delayResp_all(end+1) = delayResp;
        trial_mat2 = [];
end
% pks_trial_re = reshape(pks_trial_all,[length(pks_trial_all)/length(iscell_index),length(iscell_index)])'; 
% xlswrite('mean pks by trial P6_2.xlsx',pks_trial_re);
% pks_pos_re = reshape(pks_pos_all,[length(pks_pos_all)/length(iscell_index),length(iscell_index)])';
% xlswrite('pks position by trial P6_1.xlsx',pks_pos_re); % calculated in second
% delayResp_re = reshape(delayResp_all,[length(delayResp_all)/length(iscell_index),length(iscell_index)])';
% xlswrite('signal 1s-delay by trial P6_2.xlsx',delayResp_re);

puff_merge3 = {};
for p = 1:length(puff_merge2)
    trial_mat3 = puff_merge2{p};
    trial_mat4 = mean(trial_mat3,2);
    puff_merge3{end+1} = trial_mat4;
end

ind = length(puff_merge3)/length(iscell_index);
for q = 1:ind
    ACC1_all = [puff_merge3{1:ind:length(puff_merge3)}];
    ACC2_all = [puff_merge3{2:ind:length(puff_merge3)}];
    ACC3_all = [puff_merge3{3:ind:length(puff_merge3)}];
    ACC4_all = [puff_merge3{4:ind:length(puff_merge3)}];
    ACC5_all = [puff_merge3{5:ind:length(puff_merge3)}];
    ACC6_all = [puff_merge3{6:ind:length(puff_merge3)}];
    SAT1_all = [puff_merge3{7:ind:length(puff_merge3)}];
    SAT2_all = [puff_merge3{8:ind:length(puff_merge3)}];
    SAT3_all = [puff_merge3{9:ind:length(puff_merge3)}];
    SAT4_all = [puff_merge3{10:ind:length(puff_merge3)}];
    SAT5_all = [puff_merge3{11:ind:length(puff_merge3)}];
    SAT6_all = [puff_merge3{12:ind:length(puff_merge3)}];
    SAT7_all = [puff_merge3{13:ind:length(puff_merge3)}];
    SAT8_all = [puff_merge3{14:ind:length(puff_merge3)}];
    SAT9_all = [puff_merge3{15:ind:length(puff_merge3)}];
    SAT10_all = [puff_merge3{16:ind:length(puff_merge3)}];

end

puff_sheet = {ACC1_all,ACC2_all,ACC3_all,ACC4_all,ACC5_all,ACC6_all,SAT1_all,SAT2_all,SAT3_all,SAT4_all,SAT5_all,SAT6_all,SAT7_all,SAT8_all,SAT9_all,SAT10_all};

puff_sheet2 = {};
for w = 1:length(puff_sheet)
    mat_day = puff_sheet{w};
    mat_day2 = mat_day(47:64,:);
    mat_day3 = mat_day2';
    puff_sheet2{end+1} = mat_day3;
end

% extract AUC from individual neurons
AUC_all = [];
for ext = 1:length(puff_sheet2)
    CellsByDay = puff_sheet2{ext};
    CellSum = sum(CellsByDay(:,6:10), 2); % Sum across columns 6 to 10
    AUC_all = [AUC_all, CellSum];
end

% extract peak response from individual neurons
PeakAll = [];
for ext = 1:length(puff_sheet2)
    CellsByDay = puff_sheet2{ext};
    CellPeak = max(CellsByDay(:,6:10),[],2); %16:20
    PeakAll = [PeakAll,CellPeak];
end

% extract response trace from individual days
mean_results = cell(size(puff_sheet2));
for i = 1:numel(puff_sheet2)
    current_cell = puff_sheet2{i};
    nonnan_mean = nanmean(current_cell);
    mean_results{i} = nonnan_mean;
end
mean_results2 = zeros(length(mean_results), length(mean_results{1}));
for i = 1:length(mean_results)
    mean_results2(i, :) = mean_results{i};
end

for x = 1:length(puff_sheet2)
    xlswrite('df.f_Fneu corrected P6_3.xlsx',puff_sheet2{x},x)
end

%% dF/F0 (responsive trials only)

trial_mat2 = [];
puff_merge2 = {};
prestim_all= {};
poststim_all = {};
response2 = [];
response_all = {};
prestim_sd_all = {};
pos2 = [];
position_all = {};
baseline_raw_all = {};
for s = 1: length(puff_merge)
        for d = 1:size(puff_merge{s},2)
            baseline_raw = mean(puff_merge{s}(47:51,:)); % 3s baseline; if use 1s, change to 37:51
            trial_mat = puff_merge{s}(:,d);
            dF_F = (trial_mat - mean(trial_mat(47:51)))/mean(trial_mat(47:51));% 3s baseline; if use 1s, change to 47:51
            trial_mat2 = [trial_mat2,dF_F];
            prestim = trial_mat2(47:51,:); % 3s baseline; if use 1s, change to 47:51
            prestim_base = mean(prestim);
            prestim_sd = std(prestim);
            poststim = trial_mat2(52:56,:);
            [poststim_base,position] = max(poststim);

            for e = 1:length(prestim_base)
                if poststim_base(e) > prestim_base(e)+2*prestim_sd(e)
                    response = 1;
                    pos = position(e);
                else
                    response = 0;
                    pos = 0;
                end
            end
            response2 = [response2,response];
            pos2 = [pos2,pos];
        end
        puff_merge2{end+1} = trial_mat2;
        prestim_all{end+1} = prestim_base;
        prestim_sd_all{end+1} = prestim_sd;
        poststim_all{end+1} = poststim_base;
        response_all{end+1} = response2;
        position_all{end+1} = pos2;
        baseline_raw_all{end+1} = baseline_raw;
        trial_mat2 = [];
        prestim_base = [];
        poststim_base = [];
        response2 = [];   
        pos2 = [];
        baseline_raw2=[];
end

perctrial_all = [];
for num = 1:length(response_all)
    percent_trial = sum(response_all{num})/length(response_all{num});
    perctrial_all = [perctrial_all,percent_trial];
end
perctrial_re = reshape(perctrial_all,[imaging_day/2,length(response_all)/(imaging_day/2)])';
xlswrite('responsive trials percent_P6_3.xlsx',perctrial_re);

resp_trial_all = {};
for num2 = 1:length(puff_merge2)
    resp_mat=[];
    for num3 = 1:length(response_all{num2})
        if response_all{num2}(num3)==1   %change between 1 and 0 to decide responsive / suppressed trials
            resp_trial = puff_merge2{num2}(:,num3);
        elseif response_all{num2}(num3)==0
               resp_trial = NaN(size(puff_merge2{num2}(:,num3)));
        end
        resp_mat = [resp_mat,resp_trial];
    end
    resp_trial_all{end+1} = resp_mat;
end

puff_merge4 = {};
for p = 1:length(resp_trial_all)
    trial_mat5 = resp_trial_all{p};
    trial_mat6 = mean(trial_mat5,2,'omitnan');
    puff_merge4{end+1} = trial_mat6;
end

ind = length(puff_merge4)/length(iscell_index);
for q = 1:ind
    ACC1_all = [puff_merge4{1:ind:length(puff_merge4)}];
    ACC2_all = [puff_merge4{2:ind:length(puff_merge4)}];
    ACC3_all = [puff_merge4{3:ind:length(puff_merge4)}];
    ACC4_all = [puff_merge4{4:ind:length(puff_merge4)}];
    ACC5_all = [puff_merge4{5:ind:length(puff_merge4)}];
    ACC6_all = [puff_merge4{6:ind:length(puff_merge4)}];
    SAT1_all = [puff_merge4{7:ind:length(puff_merge4)}];
    SAT2_all = [puff_merge4{8:ind:length(puff_merge4)}];
    SAT3_all = [puff_merge4{9:ind:length(puff_merge4)}];
    SAT4_all = [puff_merge4{10:ind:length(puff_merge4)}];
    SAT5_all = [puff_merge4{11:ind:length(puff_merge4)}];
    SAT6_all = [puff_merge4{12:ind:length(puff_merge4)}];
    SAT7_all = [puff_merge4{13:ind:length(puff_merge4)}];
    SAT8_all = [puff_merge4{14:ind:length(puff_merge4)}];
    SAT9_all = [puff_merge4{15:ind:length(puff_merge4)}];
    SAT10_all = [puff_merge4{16:ind:length(puff_merge4)}];

end

puff_sheet = {ACC1_all,ACC2_all,ACC3_all,ACC4_all,ACC5_all,ACC6_all,SAT1_all,SAT2_all,SAT3_all,SAT4_all,SAT5_all,SAT6_all,SAT7_all,SAT8_all,SAT9_all,SAT10_all};

puff_sheet3 = {};
for w = 1:length(puff_sheet)
    mat_day = puff_sheet{w};
    mat_day2 = mat_day(47:64,:);
    mat_day3 = mat_day2';
    puff_sheet3{end+1} = mat_day3;
end

for x = 1:length(puff_sheet3)
    xlswrite('responsive trials only_P6_3.xlsx',puff_sheet3{x},x)
end

% extract peak response from individual neurons
PeakAll = [];
for ext = 1:length(puff_sheet3)
    CellsByDay = puff_sheet3{ext};
    CellPeak = max(CellsByDay(:,6:10),[],2); %16:20
    PeakAll = [PeakAll,CellPeak];
end


% extract response trace from individual days
mean_results = cell(size(puff_sheet3));
for i = 1:numel(puff_sheet3)
    current_cell = puff_sheet3{i};
    nonnan_mean = nanmean(current_cell);
    mean_results{i} = nonnan_mean;
end
mean_results2 = zeros(length(mean_results), length(mean_results{1}));
for i = 1:length(mean_results)
    mean_results2(i, :) = mean_results{i};
end


%% Spontaneous activity for training dataset

% Spon activity before stimulus train

event_num = [];
event_num3 = [];
event_all = {};
event_all3 = {};
pks_amp_ave = [];
pks_amp_all = {};
pksAUV_ave = [];
pksAUV_all = {};
for x = 1:length(base_per_day)  
    for e = 1:size(base_per_day{x},2)
        base_mat2 = base_per_day{x};
        base_mat = base_mat2(1:459,:);
        base_mat3 = base_mat2(end-459:end,:);
        pks_cell = findpeaks(zscore(base_mat(:,e)), MinPeakDistance=4);
        pks_cell3 = findpeaks(zscore(base_mat3(:,e)), MinPeakDistance=4);
        event_cell = length(find(pks_cell>nanmean(zscore(base_mat(:,e)))+2*nanstd(zscore(base_mat(:,e)))));
        event_cell3 = length(find(pks_cell3>nanmean(zscore(base_mat3(:,e)))+2*nanstd(zscore(base_mat3(:,e)))));
        pks_amp = nanmean(pks_cell(find(pks_cell>nanmean(zscore(base_mat(:,e)))+2*nanstd(zscore(base_mat(:,e))))));
        
        if length(find(pks_cell>nanmean(zscore(base_mat(:,e)))+2*nanstd(zscore(base_mat(:,e)))))~=1
            pksAUV = trapz(find(pks_cell>nanmean(zscore(base_mat(:,e)))+2*nanstd(zscore(base_mat(:,e)))),pks_cell(find(pks_cell>nanmean(zscore(base_mat(:,e)))+2*nanstd(zscore(base_mat(:,e))))));
        else
            pksAUV = pks_cell(find(pks_cell>nanmean(zscore(base_mat(:,e)))+2*nanstd(zscore(base_mat(:,e)))));
        end
        
        pksAUV_ave(end+1) = pksAUV;
        pks_amp_ave(end+1) = pks_amp;
        event_num(end+1) = event_cell;
        event_num3(end+1) = event_cell3;
%         event_normed = event_num/event_num(1);
    end
    event_all{end+1} = event_num;
    event_all3{end+1} = event_num3;
    pks_amp_all{end+1} = pks_amp_ave;
    pksAUV_all{end+1} = pksAUV_ave;
    event_num = [];
    event_num3 = [];
    pks_amp_ave = [];
    pksAUV_ave = [];
end

event_mat = reshape(cell2mat(event_all),[length(cell2mat(event_all))/length(event_all),length(event_all)])';
event_mat3 = reshape(cell2mat(event_all3),[length(cell2mat(event_all3))/length(event_all3),length(event_all3)])';
pks_mat = reshape(cell2mat(pks_amp_all),[length(cell2mat(pks_amp_all))/length(pks_amp_all),length(pks_amp_all)])';
pksAUV_mat = reshape(cell2mat(pksAUV_all),[length(cell2mat(pksAUV_all))/length(pksAUV_all),length(pksAUV_all)])';

event_mat4 = [];
for a = 1:size(event_mat,1)
    event_cell = reshape(event_mat(a,:),2,size(event_mat,2)/2);
    event_day = nansum(event_cell);
    event_mat4 = [event_mat4;event_day];
end

event_mat5 = [];
for a = 1:size(event_mat,1)
    event_cell3 = reshape(event_mat3(a,:),2,size(event_mat3,2)/2);
    event_day3 = nansum(event_cell3);
    event_mat5 = [event_mat5;event_day3];
end

pks_mat2 = [];
for a = 1:size(pks_mat,1)
    amp_cell = reshape(pks_mat(a,:),2,size(pks_mat,2)/2);
    pks_day = nansum(amp_cell);
    pks_mat2 = [pks_mat2;pks_day];
end

pksAUV_mat2 = [];
for a = 1:size(pksAUV_mat,1)
    AUV_cell = reshape(pksAUV_mat(a,:),2,size(pksAUV_mat,2)/2);
    pksAUV_day = nansum(AUV_cell);
    pksAUV_mat2 = [pksAUV_mat2;pksAUV_day];
end

event_normed_all = [];
for a = 1:size(event_mat,1)
    event_cell = reshape(event_mat(a,:),2,size(event_mat,2)/2);
    event_day = sum(event_cell);
    event_normed = event_day/mean(event_day(2:6)); %change baseline days: ACC2-6
    event_normed_all = [event_normed_all;event_normed];
    event_normed = [];
end
event_normed_all3 = [];
for a = 1:size(event_mat3,1)
    event_cell3 = reshape(event_mat3(a,:),2,size(event_mat3,2)/2);
    event_day3 = sum(event_cell3);
    event_normed3 = event_day3/mean(event_day3(2:6)); %change baseline days: ACC2-6
    event_normed_all3 = [event_normed_all3;event_normed3];
    event_normed3 = [];
end

xlswrite('spontaneous activity pre_P6_1.xlsx',event_normed_all);
xlswrite('spontaneous activity post_P6_1.xlsx',event_normed_all3);

xlswrite('spon activity pre zscored_nonnormed P6_1.xlsx',event_mat4);
xlswrite('spon activity post zscored_nonnormed P6_2.xlsx',event_mat5);
xlswrite('spon activity ave peak pre P6_2.xlsx',pks_mat2);
xlswrite('spon activity AUV pre P6_2.xlsx',pksAUV_mat2);
result = event_mat4 / (459/5.11);


% Spon activity during stimulus train

event_freq_all = [];
between_amp = [];
puff_rmv_event_all=[];
for num8 = 1:length(puff_merge)
    puff_rmv = puff_merge{num8};
    puff_rmv(50:80,:)  = [];  % remove signals from 2s (52-61) response time window
    puff_rmv2 = puff_rmv(:);
    puff_rmv_pks = findpeaks(zscore(puff_rmv2), MinPeakDistance=4);
    [pks, pos] = findpeaks(zscore(puff_rmv2), MinPeakDistance=4);
    xs = 1:length(puff_rmv2);
    plot(xs, puff_rmv2, xs(pos), puff_rmv2(pos), "o");
   
    puff_rmv_event = length(find(puff_rmv_pks>nanmean(zscore(puff_rmv2))+2*nanstd(zscore(puff_rmv2))));
    puff_rmv_amp = nanmean(puff_rmv_pks(find(puff_rmv_pks>nanmean(zscore(puff_rmv2))+2*nanstd(zscore(puff_rmv2)))));
    
    new_pos = pos(find(puff_rmv_pks>nanmean(zscore(puff_rmv2))+2*nanstd(zscore(puff_rmv2))));
    plot(xs, puff_rmv2, xs(new_pos), puff_rmv2(new_pos), "o");
    
    event_freq = puff_rmv_event/(size(puff_rmv,2)*((102/5.11)-(5/5.11)));
    puff_rmv_event_all(end+1) = puff_rmv_event;
    event_freq_all(end+1) = event_freq;
    between_amp(end+1) = puff_rmv_amp;
end

event_freq_all2 = [];
between_amp2 = [];
blank_rmv_event_all=[];
for num9 = 1:length(blank_merge)
    blank_rmv = blank_merge{num9};
    blank_rmv(50:80,:)  = [];  % remove signals from 2s (52-61) response time window
    blank_rmv2 = blank_rmv(:);
    blank_rmv_pks = findpeaks(zscore(blank_rmv2), MinPeakDistance=4);
    
    [pks, pos] = findpeaks(zscore(blank_rmv2), MinPeakDistance=4);
    xs = 1:length(blank_rmv2);
    plot(xs, blank_rmv2, xs(pos), blank_rmv2(pos), "o");
    
    blank_rmv_event = length(find(blank_rmv_pks>nanmean(zscore(blank_rmv2))+2*nanstd(zscore(blank_rmv2))));
    blank_rmv_amp = nanmean(blank_rmv_pks(find(blank_rmv_pks>nanmean(zscore(blank_rmv2))+2*nanstd(zscore(blank_rmv2)))));
    
    new_pos = pos(find(blank_rmv_pks>nanmean(zscore(blank_rmv2))+2*nanstd(zscore(blank_rmv2))));
    
    event_freq2 = blank_rmv_event/(size(blank_rmv,2)*((102/5.11)-(5/5.11)));
    blank_rmv_event_all(end+1) = blank_rmv_event;
    event_freq_all2(end+1) = event_freq2;
    between_amp2(end+1) = blank_rmv_amp;
end

rmv_event_all=puff_rmv_event_all+blank_rmv_event_all;
event_freq_all3 = rmv_event_all/((length(puff_rmv2)+length(blank_rmv2))/5.11);
between_amp3 = nanmean([between_amp;between_amp2]);

spon_during = reshape(event_freq_all3,[imaging_day/2, length(iscell_index)])';
spon_amp_during = reshape(between_amp3,[imaging_day/2, length(iscell_index)])';
xlswrite('spon activity during zscored_freq P6_2.xlsx',spon_during);
xlswrite('spon activity during_amp P6_2.xlsx',spon_amp_during);





