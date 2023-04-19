%% pre-stimulus power analysis using dothewave.m (MorletWavelets)
% author: zeguo.qiu@uq.net.au
% last date modified: 06/01/2023

%% pre-processing, power extraction and single-trial regression
addpath 'Z:\FreqAnalysis';
addpath 'Z:\eeglab2021.1';
group_path = 'Z:\Experiment_Search\EEGsets\';
analysistype = 1; % 1-awareness overall regardless of emotion; 2-awareness and emotion
eeglab
chsum=[16 20:29];
tsum=251:500;
PIDsum=[1:25 27 28 30:32];
allPs_awareness=zeros(27,250,30);
allPs_emotion=zeros(27,250,30);
for PID=[1:25 27 28 30:32]
    pow=[];
    comp=[];
    % Load data to extract epoch information
    EEG1 = pop_loadset('filename',[num2str(PID) 'AR.set'], 'filepath', [group_path  num2str(PID) '\']); % load dataset
    epochNo=[];
    for r=1:length(EEG1.epoch)
        epochNo_r=cell2mat(EEG1.epoch(r).eventbepoch(1,1));
        epochNo=[epochNo epochNo_r];
    end
    % Load data to run power analysis
    EEG = pop_loadset('filename',[num2str(PID) 'ICA.set'], 'filepath', [group_path  num2str(PID) '\']); % load dataset
    EEG = pop_creabasiceventlist( EEG , 'AlphanumericCleaning', 'on', 'BoundaryNumeric', { -99 }, 'BoundaryString', { 'boundary' }); %create eventlist
    EEG = pop_binlister( EEG , 'BDF', [group_path 'Bin_descriptor_Search.txt'], 'IndexEL',  1, 'SendEL2',...
     'EEG', 'UpdateEEG', 'on', 'Voutput', 'EEG' ); %identify and categorise striggers into bins
    EEG = pop_epochbin( EEG , [-1000.0  1000.0],  [ -200 0]); %cut data into epochs [-1s, 1s], baseline correction using -200 till stim onset
    EEG = pop_select( EEG, 'trial',epochNo ); %select clean epochs
    removeIndex=[];
    for r=1:length(EEG.epoch) % contrast unaware and multifix-aware using bins 1,2,6,7; contrast unaware and first-sight aware using bins 1,4,6,9
        if EEG.epoch(r).eventbini{1,find(cell2mat(EEG.epoch(r).eventbini)~=-1,1)}~=1 && EEG.epoch(r).eventbini{1,find(cell2mat(EEG.epoch(r).eventbini)~=-1,1)}~=4 && EEG.epoch(r).eventbini{1,find(cell2mat(EEG.epoch(r).eventbini)~=-1,1)}~=6 && EEG.epoch(r).eventbini{1,find(cell2mat(EEG.epoch(r).eventbini)~=-1,1)}~=9
            removeIndex=[removeIndex r];
        end
    end
    EEG = pop_select( EEG, 'notrial',removeIndex );
    % Extract power
    [pow, phase, comp, dstimes, freqs] = dothewave(EEG.data, EEG.srate, [3 30],...
        27, [3 8], 1, [], 'ms');
    % Create design matrix
    if analysistype==1
        bin=zeros(EEG.trials,1);
    else
        bin=zeros(EEG.trials,2);
    end
    % contrast unaware and multifix-aware using bins 1,2,6,7; for
    % first-sight aware use bins 1,4,6,9
    for n=1:EEG.trials
        bin(n,1)=EEG.epoch(n).eventbini{1,find(cell2mat(EEG.epoch(n).eventbini)~=-1,1)};
        if analysistype == 1
            if bin(n,1)==6; bin(n,1)=1; elseif bin(n,1)==4; bin(n,1)=2; elseif bin(n,1)==9; bin(n,1)=2; end
        elseif analysistype == 2
            if bin(n,1)==6; bin(n,1)=1;bin(n,2)=2; elseif bin(n,1)==9; bin(n,1)=2;bin(n,2)=2;
            else
                bin(n,2)=1;
            end
        end
    end
    C=[ones(EEG.trials,1) bin];
    % Calculate regression coefficients
    D=zeros(EEG.trials,1);
    betaperm_awareness=zeros(2000,1);
    betaperm_emotion=zeros(2000,1);
    beta_awareness_z=zeros(27,250,11);
    beta_emotion_z=zeros(27,250,11);
    tic
    for ch=[16 20:29]       %posterior electrodes are [16 20:29]
        for freq=1:27       %this number depends on the frequency range
            for t=251:500   %t=251:501 is time window of [-500 0]; can change this time windows for post-stimulus signal analyses
                for trial=1:EEG.trials
                    D(trial,1)=pow(ch,freq,t,trial);
                end
                [~,p] = sort(D,'descend');
                [~,R] = sort(p); %as a result, the larger the R, the smaller the power
                beta=(inv(C'*C))*C'*R; %calculate beta using rank scored D (i.e., R); the following permutation is also operated on vector R
                for permutate=1:2000
                    Rperm=R(randperm(length(R)));
                    tmp_betaperm=(inv(C'*C))*C'*Rperm;
                    betaperm_awareness(permutate,1)=tmp_betaperm(2,1);
                    if analysistype == 2
                        betaperm_emotion(permutate,1)=tmp_betaperm(3,1);
                    end
                end
                betaperm_awareness_mean=mean(betaperm_awareness);betaperm_awareness_std=std(betaperm_awareness);
                beta_awareness_z(freq,find(t==tsum),find(chsum==ch))=(beta(2,1)-betaperm_awareness_mean)/betaperm_awareness_std;
                if analysistype == 2
                    betaperm_emotion_mean=mean(betaperm_emotion);betaperm_emotion_std=std(betaperm_emotion);
                    beta_emotion_z(freq,find(t==tsum),find(chsum==ch))=(beta(3,1)-betaperm_emotion_mean)/betaperm_emotion_std;
                end
            end
        end
    end
    toc
    posterior_Fawareness=mean(beta_awareness_z(:,:,:),3);
    writematrix(posterior_Fawareness,['P' num2str(PID) '_awareness.txt'],'Delimiter','tab'); %save as temporary text files
    if analysistype == 2
        posterior_emotion=mean(beta_emotion_z(:,:,:),3);
    	writematrix(posterior_emotion,['P' num2str(PID) '_emotion.txt'],'Delimiter','tab');
    end
    %tmpSaved = readmatrix('P1.txt');
    allPs_awareness(:,:,find(PIDsum==PID))=posterior_Fawareness;
    %allPs_emotion(:,:,find(PIDsum==PID))=posterior_emotion;
end

for PID=[1:25 27 28 30:32]
    tmpSaved_awareness = readmatrix(['P' num2str(PID) '_awareness.txt']);
    allPs_awareness(:,:,find(PIDsum==PID))=tmpSaved_awareness;
%     tmpSaved_emotion = readmatrix(['P' num2str(PID) '_emotion.txt']);
%     allPs_emotion(:,:,find(PIDsum==PID))=tmpSaved_emotion;
end%preparing data for significance tests

%% group-level significance testing on betas using the mxt_perm1.m from MassUnivariateAnalysis Toolbox
% dataset should include data from all Ps (filename is allPs)
[pval, t_orig, tmx_ptile, est_alpha]=mxt_perm1(allPs_awareness,2000,.05,0,1);
