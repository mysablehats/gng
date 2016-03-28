function [nodes, edges,s1,s2] = gng_lax(Data,PARAMS)
if (isempty(PARAMS.nodes)||PARAMS.nodes==0||isempty(Data))
    error('Wrong input arguments. Either .nodes or Data is zero, or empty.')
end
% Unsupervised Self Organizing Map. Growing Neural Gas (GNG) Algorithm.

% Main paper used for development of this neural network was:
% Fritzke B. "A Growing Neural Gas Network Learns Topologies", in 
%                         Advances in Neural Information Processing Systems, MIT Press, Cambridge MA, 1995.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumOfEpochs   = 5;
NumOfSamples = fix(size(Data,2)/NumOfEpochs);
age_inc               = 1;
max_age             = PARAMS.amax;
max_nodes         = PARAMS.nodes;
eb                         = PARAMS.eb;
en                         = PARAMS.en;
lamda                   = PARAMS.lambda;%3;
alpha                    = PARAMS.alpha;%.5;     % q and f units error reduction constant.
d                           = PARAMS.d;%.99;   % Error reduction factor.
RMSE                  = zeros(1,NumOfEpochs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PLOTIT = PARAMS.PLOTIT;%= false;
RANDOMSTART = PARAMS.RANDOMSTART;


% Define the params vector where the GNG algorithm parameters are stored:
params = [ age_inc;
                    max_age;
                    max_nodes;
                    eb;
                    en;
                    alpha;
                    lamda;
                    d;
                    0;   % Here, we insert the sample counter.
                    1;   % This is reserved for s1 (bmu);
                    2;]; % This is reserved for s2 (secbmu);

Cur_RMSE = zeros(1,NumOfEpochs);
RMSE = [];
Epoch = [];
Cur_NumOfNodes = [];

% Step.0 Start with two neural units (nodes) selected from input data:
NumOfNodes = 2;

ni1 = 1; 
ni2 = 2; 
if RANDOMSTART
    nn = randperm(size(Data,2),2);
    ni1 = nn(1);
    ni2 = nn(2);
elseif isfield(PARAMS,'startingpoint')&&~isempty('PARAMS.startingpoint')
    ni1 = PARAMS.startingpoint(1);
    ni2 = PARAMS.startingpoint(2);
end
n1 = Data(:,ni1); n2 = Data(:,ni2);

nodes = [n1 n2];



% Initial connections (edges) matrix.
edges = [0  1;
                1  0;];
     
% Initial ages matrix.
ages = [ NaN  0;
               0  NaN;];

% Initial Error Vector.
errorvector = [0 0];

% scrsz = get(0,'ScreenSize');
% figure('Position',[scrsz(3)/2 scrsz(4)/3-50 scrsz(3)/2 2*scrsz(4)/3])


%%%%%%%%%%MESSAGES PART
dbgmsg('generates GNG A and C matrices',1)
dbgmsg('Executing GNG with: ', num2str(PARAMS.nodes),' nodes.',1)
dbgmsg(num2str(params),1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for kk=1:NumOfEpochs
    
% Choose the next Input Training Vectors.
nextblock = (kk-1)*NumOfSamples+1:1:kk*NumOfSamples;

% Step.1 Generate an input signal � according to P(�).
In = Data(:,nextblock);

for n=1:NumOfSamples

params(9) = n;

% Step 2. Find the two nearest units s1 and s2 to the new data sample.
[s1, s2, distances] = findTwoNearest(In(:,n),nodes);
params(10) = s1;
params(11) = s2;

% Steps 3-6. Increment the age of all edges emanating from s1 .
[nodes, edges, ages, errorvector] = edgeManagement(In(:,n),nodes,edges,ages,errorvector, distances, params);

% Step 7. Dead Node Removal Procedure.
[nodes, edges, ages, errorvector] = removeUnconnected(nodes, edges,ages,errorvector);

% Step 8. Node Insertion Procedure.
if mod(n,lamda)==0 && size(nodes,2)<max_nodes
     [nodes,edges,ages,errorvector] = addNewNeuron(nodes,edges,ages,errorvector,alpha);
end
    
% Step 9. Finally, decrease the error of all units.
errorvector = d*errorvector;

end

if PLOTIT
    NumOfNodes = size(nodes,2);
    Cur_NumOfNodes = [Cur_NumOfNodes NumOfNodes];
    if length(Cur_NumOfNodes)>100
        Cur_NumOfNodes = Cur_NumOfNodes(end-100:end);
    end

    Cur_RMSE(kk) = norm(errorvector)/sqrt(NumOfNodes);
    RMSE = [RMSE Cur_RMSE(kk)];
    if length(RMSE)>100
        RMSE = RMSE(end-100:end);
    end

    Epoch = [Epoch kk];
    if length(Epoch)>100
        Epoch = Epoch(end-100:end);
    end
    subplot(1,2,1);
    plotgwr(nodes,edges);%,'n'); %improved to show snazzy skeletors
    % xlim([-1/2 2.5]);
    % ylim([-1 8]);
    % zlim([-1/2 1.5]);
    % xlim([-1 6]);
    % ylim([-1 6]);
    % zlim([-7 7]);
    drawnow;

    subplot(2,2,2);
    plot(Epoch,RMSE,'r.');
    title('RMS Error');
    if kk>100
         xlim([Epoch(1) Epoch(end)]);
    end
    xlabel('Training Epoch Number');
    grid on;

    subplot(2,2,4);
    plot(Epoch,Cur_NumOfNodes,'g.');
    title('Number of Neural Units in the Growing Neural Gas');
    if kk>100
      xlim([Epoch(1) Epoch(end)]);
    end
    xlabel('Training Epoch Number');
    grid on;

end

end
dbgmsg(strcat('End number of nodes:',num2str(size(nodes,2)),' With MAXNODES:',num2str(PARAMS.nodes)),1)
end