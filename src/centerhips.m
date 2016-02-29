function newskel = centerhips(skel)
%%%%%%%%%MESSAGES PART
%%%%%%%%ATTENTION: this function is executed in loops, so running it will
%%%%%%%%messages on will cause unpredictable behaviour
%dbgmsg('Removing displacement based on hip coordinates (1st point on 25x3 skeleton matrix) from every other')
%dbgmsg('This makes the dataset translation invariant')
%%%%%%%%%%%%%%%%%%%%%
hh = size(skel,1)/3;
%%%% reshape skeleton
if all(size(skel) == [75 1]) % checks if the skeleton is a 75x1
    tdskel = zeros(25,3);
    for i=1:3
        for j=1:25
            tdskel(j,i) = skel(j+25*(i-1));
        end
    end
elseif all(size(skel) == [150 1])
    tdskel = reshape(skel,hh,3); %%%%%I think this should work for every skeleton Nx3, but I will not change the function that came before, because (at least I think), it is working
elseif all(size(skel) == [25 3])
        tdskel = skel;
else
    error('Do not know this size of skeleton yet!')
end

hips = [repmat(tdskel(1,:),25,1);zeros(hh-25,3)]; 

newskel = tdskel - hips;
newskel(1,:) = [];
% newskel = zeros(size(tdskel)-[1 0]);
% for i = 2:hh
%     newskel(i-1,:) = tdskel(i,:)- 1*hips;
% end

%I need to shape it back into 75(-3 now) x 1
newskel = [newskel(:,1);newskel(:,2);newskel(:,3)]; % I think....
if ~(all(size(newskel) == [72 1])||all(size(newskel) == [147 1]))
    error('wrong skeleton size!')
end