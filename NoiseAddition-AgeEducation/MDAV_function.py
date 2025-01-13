import pandas as pd


# def MDAVwMSE(NumAtDimAtX,K):
# %k-Anonymous microaggregation of a numerical dataset, assuming MSE as distortion measure.
# %
# %Implementation simpler yet completely equivalent to pseudocode given as Algorithm 5.1, MDAV-generic, in  
# %   J. Domingo-Ferrer and V. Torra, "Ordinal, continuous and heterogeneous k-anonymity through microaggregation," 
# %   Data Min., Knowl. Disc., vol. 11, no. 2, pp. 195-212, 2005.
# %w/ mean, known to minimize MSE, as centroid
# %
# %Find centroid, find furthest point P from centroid, find furthest point Q from P. 
# %Group k-1 nearest points from P into a group, and do the same with Q. 
# %Repeat on the points left until there are less than 2k points. 
# %If there are k to 2k-1 points, form a group and finish.
# %If there are 1 to k-1 points, adjoin those to last (hopefully nearest) group.
    
# %MDAV
# %Indices of ungrouped samples, all initially
# Is=(1:XAmt)'; IAmt=XAmt; Q=1;

# %While there are ENOUGH SAMPLES for at least two groups
# while IAmt>=2*K 
#     %Initially, X0 is the CENTROID of the ungrouped samples
#     X0=sum(NumAtDimAtX(:,Is),2)/IAmt; 
#     %Precompute distances to X0
#     %Ds=sum((NumAtDimAtX(:,Is)-repmat(X0,1,IAmt)).^2,1);
#     %Looping through the dimensions is faster than using repmat
#     %Ds=0; for Dim=1:DimAmt, Ds=Ds+(NumAtDimAtX(Dim,Is)-X0(Dim)).^2; end
#     %But singleton expansion is faster than looping through the dimensions
#     Ds=sum(bsxfun(@minus,NumAtDimAtX(:,Is),X0).^2,1);
    
#     for G=1:2 %Two groups are made
#         %Initially, X0 is the centroid and X1 is P; later, X0 will be P and X1 will be Q
#         %Find FURTHEST point X1 from X0
#         [~,IFar]=max(Ds); X1=NumAtDimAtX(:,Is(IFar)); 
        
#         %Group NEAREST k-1 points from X1
#         %[~,IsSort]=sort(sum((NumAtDimAtX(:,Is)-repmat(X1,1,IAmt)).^2,1)); IsSort=IsSort(1:K);
#         %Ds=0; for Dim=1:DimAmt, Ds=Ds+(NumAtDimAtX(Dim,Is)-X1(Dim)).^2; end
#         Ds=sum(bsxfun(@minus,NumAtDimAtX(:,Is),X1).^2,1);
#         %Naive implementation for partial sort; quickselect algorithm would be faster
#         [~,IsSort]=sort(Ds); IsSort=IsSort(1:K);
#         XsSort=Is(IsSort); QAtX(XsSort)=Q;
        
#         %REMOVE samples already aggregated
#         Is(IsSort)=[]; IAmt=IAmt-K; %IAmt=size(Is,1); %Is=find(QAtX==0) is slower
        
#         %In the NEXT iteration, X0 becomes P and X1 becomes (point) Q
#         Ds(IsSort)=[]; Q=Q+1;
#     end
# end

# %There are NOT ENOUGH SAMPLES for two groups anymore (1 to 2k-1 points left)
# %Either enough samples for a new group (between k and 2k-1 points not yet quantized), or 
# %not even enough samples for a new group (at most k-1 points left), join to last, (hopefully nearest) group
# QAtX(Is)=QAmt;

# %pQAtQ=[ repmat(K,QAmt-1,1) ;  XAmt-K*(QAmt-1) ]/XAmt;
# nQAtQ=[ repmat(uint32(K),QAmt-1,1) ;  uint32(XAmt-K*(QAmt-1)) ];
 
    
    
    
    # return QAtX,QAmt,nQAtQ


def pandasMDAVwMSE(path_of_data: str, column_name: str, k: int, Change_value_for_the_centroid=True):
    df=pd.read_csv(path_of_data)
    # Normalize the data
    principal=(df[column_name]-df[column_name].mean())/df[column_name].std()
    df_list=[]
    
    while principal.shape[0]>=2*k:   
        centroid=principal.mean()
        # xr
        xr_df=(principal-centroid)**2
        list_min_xr=xr_df.sort_values()
        xr=list_min_xr.iloc[0]
        xr_kminus1=list_min_xr.iloc[k]
        xr_cluster=list_min_xr.iloc[0:k]
        index_xr=xr_cluster.index
        #xs
        xs=list_min_xr.iloc[-1]
        xs_cluster= list_min_xr.iloc[-k: 0] ##Aquí creo que está mal. El mecanismo hace el clustering de los k puntos más cercanos al punto más alejado del centroide, no los k puntos más alejados del centroide
        index_xs=xs_cluster.index
        #Eliminate xr and xs registers 
        principal=principal.drop(index=index_xr)
        principal=principal.drop(index=index_xs)
        df_list.append(xr_cluster)
        df_list.append(xs_cluster)
    if (principal.shape[0]>=k) and (principal.shape[0]<=2*k-1):
        df_list.append(principal)
    if principal.shape[0]<k:
        last_cluster=pd.DataFrame(df_list[-1])
        beforelast_cluster=pd.DataFrame(df_list[-2])
        
        lc=last_cluster.iloc[0]
        blc=beforelast_cluster.iloc[-1]
    
        last_cluster_dist=(principal-lc)**2
        beforelast_cluster_dist=(principal-blc)**2

        indexlast_cluster=last_cluster_dist.index.tolist()
        indexbeforelast_cluster_dist=beforelast_cluster_dist.index.tolist()
        # print((last_cluster_dist<beforelast_cluster_dist).index.tolist())
        for i in range(principal.shape[0]):
            # b=abs(principal.iloc[i] -lc)
            # print("b ", b)
            if (last_cluster_dist.iat[i] <= beforelast_cluster_dist.iat[i]):
                a=pd.DataFrame(data=principal.iat[i], index=indexlast_cluster[i])
                last_cluster=pd.concat([last_cluster, a])
            elif(last_cluster_dist.iat[i] > beforelast_cluster_dist.iat[i]):
                b=pd.DataFrame(data=principal.iat[i], index=indexbeforelast_cluster_dist[i])
                beforelast_cluster=pd.concat([beforelast_cluster, b])
                ##Pregunta: Esto salva el cluster en el vector df_list?
    #Change value for the centroid (the cluster mean) if True
    if (Change_value_for_the_centroid==True):
        for i in range(len(df_list)):
            df=df_list[i]
            mean=df.mean()
            df.loc[:]=mean
            df_list[i]=df
        return df_list
    else:
        return df_list
    
def Convertlist_to_dataframe(listofdataframe: list, sort_index=False):
    df_mod=pd.DataFrame(listofdataframe[0])
    for i in range(1, len(listofdataframe)):
        df_temp=pd.DataFrame(listofdataframe[i])
        df_mod=pd.concat([df_mod, df_temp]) 
    if sort_index==True:
        df_mod.sort_index(inplace=True)
    return df_mod

def find_error_pandasMDAVwMSE(path_of_data: str, column_name: str, k: int, ):
    df_list=pandasMDAVwMSE(path_of_data, column_name, k, Change_value_for_the_centroid=False)
    for i in range(len(df_list)):
        df=df_list[i]
        mean=df.mean()
        df_error=(df-mean).abs()
        df_list[i]=df_error
    df_mod=Convertlist_to_dataframe(df_list, sort_index=True)
    return df_mod