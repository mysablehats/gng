% create the structure to access the database and run the classification?
function [Data, vectordata, Y] = extractdata(structure)
Data = structure(1).skel;
Y = strcmp('Fall',structure(1).act)*ones(size(structure(1).skel,3),1);
for i = 2:length(structure)
    Data = cat(3, Data, structure(i).skel);
    Y = cat(1, Y, strcmp('Fall',structure(i).act)*ones(size(structure(i).skel,3),1));
end
% It will also construct data for a clustering analysis, whatever the hell
% that might be in this sense
vectordata = [Data(:,1,1); Data(:,2,1); Data(:,3,1)];
for i = 2:length(Data)
    vectordata = cat(2,vectordata, [Data(:,1,i); Data(:,2,i); Data(:,3,i)]);
end
Y = Y';