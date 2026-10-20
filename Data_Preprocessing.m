function writtenFiles = Data_Preprocessing(inputRoot, outputDir, options)
%DATA_PREPROCESSING Export synchronized matched-source data without per-window scaling.
% Requires Signal Processing Toolbox. This is a corrected export path, not a
% reconstruction of the papers' historical preprocessing.
% Example:
% Data_Preprocessing("Data", "data", SourceSynchronized=true)
% Keep all source files unchanged. Existing output files are never overwritten.
arguments
    inputRoot (1,1) string
    outputDir (1,1) string
    options.SourceSynchronized (1,1) logical = false
    options.SampleRate (1,1) double {mustBePositive,mustBeFinite} = 256
    options.WindowSeconds (1,1) double {mustBePositive,mustBeFinite} = 4
    options.StrideSeconds (1,1) double {mustBePositive,mustBeFinite} = 4
    options.MaxDurationMismatch (1,1) double {mustBeNonnegative,mustBeFinite} = 0.1
end
if ~options.SourceSynchronized
    error('Kinetics:Synchronization', ...
        'Use matched, synchronized source records and explicitly set SourceSynchronized=true.');
end
if ~isfolder(inputRoot)
    error('Kinetics:Input', 'Input directory does not exist: %s', inputRoot);
end
windowLength = options.SampleRate * options.WindowSeconds;
strideLength = options.SampleRate * options.StrideSeconds;
if windowLength ~= fix(windowLength) || strideLength ~= fix(strideLength) || strideLength > windowLength
    error('Kinetics:Window', 'Window and stride must be integer sample counts with stride <= window.');
end
if ~isfolder(outputDir), mkdir(outputDir); end
modes = ["Walk", "Run"];
writtenFiles = strings(0,1);
for mode = modes
    outputFile = fullfile(outputDir, "Data_Preprocessed_" + mode + "_v2.mat");
    if isfile(outputFile)
        error('Kinetics:Overwrite', 'Output exists; choose a new directory: %s', outputFile);
    end
end
subjects = dir(inputRoot);
subjects = subjects([subjects.isdir] & ~startsWith({subjects.name}, '.'));
for mode = modes
    names = {'data_IMU','data_Force_FP_LL','data_Force_FP_RL','data_Insole_LL','data_Insole_RL'};
    outputFile = fullfile(outputDir, "Data_Preprocessed_" + mode + "_v2.mat");
    chunks = struct();
    for k = 1:numel(names), chunks.(names{k}) = {}; end
    subinfo = []; speedinfo = []; modeinfo = []; window_start_sample = [];
    source_file = {};
    referenceLabels = {};
    for subjectIndex = 1:numel(subjects)
        subjectFolder = subjects(subjectIndex).name;
        subjectMatch = regexp(subjectFolder, '(\d+)$', 'tokens', 'once');
        if isempty(subjectMatch), continue; end
        subjectID = str2double(subjectMatch{1});
        if ~ismember(subjectID, 4:12), continue; end
        files = dir(fullfile(inputRoot, subjectFolder, mode, '*.mat'));
        for fileIndex = 1:numel(files)
            filePath = fullfile(files(fileIndex).folder, files(fileIndex).name);
            match = regexpi(files(fileIndex).name, '(walk|run)[_ -]?(\d+)\.mat$', 'tokens', 'once');
            if isempty(match)
                error('Kinetics:Speed', 'Cannot parse speed from %s; expected walk_09.mat or run_63.mat.', filePath);
            end
            if ~strcmpi(match{1}, mode)
                error('Kinetics:Mode', 'Filename locomotion disagrees with its Walk/Run directory: %s', filePath);
            end
            speed = str2double(match{2}) / 10; % Source filename convention: tenths of km/h.
            loaded = load(filePath, 'Datastr');
            if ~isfield(loaded, 'Datastr'), error('Kinetics:Schema', 'Missing Datastr in %s', filePath); end
            d = loaded.Datastr;
            % Report missing streams/fields in the context of the source file.
            try
            signals = {d.IMU.IMUData, d.Force.LeftForceData, d.Force.RightForceData, ...
                       d.Insole.FilteredLeft, d.Insole.FilteredRight};
            sourceRates = {d.IMU.IMUFrameRate, d.Force.FrameRate, d.Force.FrameRate, ...
                           d.Insole.ResampleRate, d.Insole.ResampleRate};
            labels = {cellstr(string(d.IMU.IMUDataLabel(:))), cellstr(string(d.Force.DataLabel(:))), ...
                      cellstr(string(d.Insole.DataLabel(:)))};
            catch sourceError
                error('Kinetics:Schema', 'Missing/invalid Datastr streams in %s: %s', filePath, sourceError.message);
            end
            if any(~cellfun(@(r) isnumeric(r) && isreal(r) && isscalar(r) && isfinite(r) && r > 0, sourceRates))
                error('Kinetics:Signal', 'Every source sampling rate must be a finite positive real scalar in %s.', filePath);
            end
            rates = cellfun(@double, sourceRates);
            for labelIndex = 1:numel(labels)
                channelNames = string(labels{labelIndex});
                if isempty(channelNames) || any(ismissing(channelNames)) || ...
                        any(strlength(strtrim(channelNames)) == 0) || numel(unique(channelNames)) ~= numel(channelNames)
                    error('Kinetics:Labels', 'Every channel must have a unique nonempty label in %s.', filePath);
                end
            end
            if isempty(referenceLabels)
                referenceLabels = labels;
            elseif ~isequal(referenceLabels, labels)
                error('Kinetics:Labels', 'Channel labels/order differ in %s; no channels will be silently truncated.', filePath);
            end
            expectedChannels = [numel(labels{1}), numel(labels{2}), numel(labels{2}), numel(labels{3}), numel(labels{3})];
            durations = zeros(1,numel(signals));
            for k = 1:numel(signals)
                values = signals{k};
                if ~isnumeric(values) || ~isreal(values) || ~ismatrix(values) || isempty(values) || size(values,2) ~= expectedChannels(k) || ...
                        any(~isfinite(values(:))) || ~isfinite(rates(k)) || rates(k) <= 0
                    error('Kinetics:Signal', 'Invalid dimensions, values, or sampling rate in %s (%s).', filePath, names{k});
                end
                durations(k) = size(values,1) / rates(k);
            end
            if max(durations) - min(durations) > options.MaxDurationMismatch
                error('Kinetics:Alignment', 'Record durations disagree in %s; review synchronization.', filePath);
            end
            for k = 1:numel(signals)
                [p,q] = rat(options.SampleRate / rates(k), 1e-10);
                signals{k} = resample(double(signals{k}), p, q);
            end
            commonLength = min(cellfun(@(v) size(v,1), signals));
            starts = 1:strideLength:(commonLength-windowLength+1);
            if isempty(starts), warning('Kinetics:Short', 'Skipping short record: %s', filePath); continue; end
            for k = 1:numel(signals)
                windows = zeros(numel(starts),windowLength,size(signals{k},2),'single');
                for windowIndex = 1:numel(starts)
                    range = starts(windowIndex):(starts(windowIndex)+windowLength-1);
                    windows(windowIndex,:,:) = single(signals{k}(range,:));
                end
                chunks.(names{k}){end+1} = windows;
            end
            count = numel(starts);
            subinfo = [subinfo; repmat(subjectID,count,1)]; %#ok<AGROW>
            speedinfo = [speedinfo; repmat(speed,count,1)]; %#ok<AGROW>
            modeinfo = [modeinfo; repmat(double(mode == "Run"),count,1)]; %#ok<AGROW>
            window_start_sample = [window_start_sample; starts(:)-1]; %#ok<AGROW>
            source_file = [source_file; repmat({char(fullfile(subjectFolder,mode,files(fileIndex).name))},count,1)]; %#ok<AGROW>
        end
    end
    if isempty(subinfo), warning('Kinetics:Empty', 'No usable %s records.', mode); continue; end
    output = struct();
    for k = 1:numel(names), output.(names{k}) = cat(1,chunks.(names{k}){:}); end
    output.subinfo = subinfo;
    output.speedinfo = speedinfo;
    output.modeinfo = modeinfo;
    output.window_start_sample = window_start_sample;
    output.source_file = source_file;
    output.label_IMU = referenceLabels{1};
    output.label_FP = referenceLabels{2};
    output.label_Insole = referenceLabels{3};
    % Publish only a complete export. A failed save/attribute write must not
    % leave an apparently valid final file that blocks the next attempt.
    temporaryFile = string(tempname(outputDir)) + ".mat";
    cleanupTemporary = onCleanup(@() removeTemporaryExport(temporaryFile));
    save(temporaryFile,'-struct','output','-v7.3');
    h5writeatt(temporaryFile,'/','normalization_state','source_native');
    h5writeatt(temporaryFile,'/','sample_rate_hz',options.SampleRate);
    h5writeatt(temporaryFile,'/','source_synchronized_asserted',uint8(options.SourceSynchronized));
    h5writeatt(temporaryFile,'/','unit_policy','Source units retained; verify each target against source documentation.');
    if isfile(outputFile), error('Kinetics:Overwrite', 'Output appeared during export: %s', outputFile); end
    [moved, message] = movefile(temporaryFile, outputFile);
    if ~moved, error('Kinetics:Export', 'Could not finalize export: %s', message); end
    clear cleanupTemporary
    writtenFiles(end+1,1) = outputFile; %#ok<AGROW>
    fprintf('Exported %d %s windows to %s\n',numel(subinfo),mode,outputFile);
end
if isempty(writtenFiles)
    error('Kinetics:Empty', 'No usable Walk or Run windows were exported; inspect the source layout and durations.');
end
end

function removeTemporaryExport(path)
% The path is created by tempname above and never refers to a source record.
if isfile(path), delete(path); end
end
