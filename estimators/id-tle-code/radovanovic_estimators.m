%dataPath = '/home/utilisateur/intrinsicDimensionPackage/data.txt';
%k = 20;
%X = dlmread(dataPath);

tic;

rng(42, 'twister');

theta = 0.975; % ratio of variance to preserve by PCA

dataExt = '.data';
dataDlm = ' ';

n = size(X,1);

id_mle = ones(n,1);
id_tle = zeros(n,1);
id_lcd = zeros(n,1);
id_mom = zeros(n,1);
id_ed = zeros(n,1);
id_ged = zeros(n,1);
id_lpca = zeros(n,1);

fprintf('\ncomputing %d-nearest neighbors...',k);
[idxmax,distsmax] = knnsearch(X,X,'K',k+1);
idxmax = idxmax(:,2:end); % 2:end skips first neighbor - the point itself
distsmax = distsmax(:,2:end);

warning('off'); % because of PCA and singular matrices

idx = idxmax(:,1:k);
dists = distsmax(:,1:k);

fprintf('\n\nk = %d\nquery point:',k);
for i = 1:n
    if mod(i,1000)==0, fprintf('\n%d',i); end
    KNN = X(idx(i,:),:);
    id_tle(i) = idtle(KNN,dists(i,:));
    id_mle(i) = idmle(dists(i,:)');
    id_mom(i) = idmom(dists(i,:)');
    id_ed(i) = ided(dists(i,:)');
    id_ged(i) = idged(dists(i,:)');
    id_lpca(i) = idlpca(KNN,theta);
end

csvwrite(['id_mle.csv'],id_mle);
csvwrite(['id_tle.csv'],id_tle);
csvwrite(['id_mom.csv'],id_mom);
csvwrite(['id_ed.csv'],id_ed);
csvwrite(['id_ged.csv'],id_ged);
csvwrite(['id_lpca.csv'],id_lpca);

warning('on');

fprintf('\n');

toc;