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
        centroide=principal.mean()
        # xr
        xr_df=(principal-centroide)**2
        xr=xr_df.max()
        xr_list=(principal-xr)**2
        list_near_xr=xr_list.sort_values()
        xr_k_near=list_near_xr.iloc[0:k]
        index_xr=xr_k_near.index
        xr_cluster=principal.loc[index_xr]
        # Agregate the cluster xr
        df_list.append(xr_cluster)
        #Remove samples already aggregated
        principal=principal.drop(index=index_xr)
        #xs
        xs=list_near_xr.iloc[-1]
        xs_list= (principal-xs)**2
        list_near_xs=xs_list.sort_values()
        xs_k_near=list_near_xs.iloc[0:k]
        index_xs=xs_k_near.index
        xs_cluster=principal.loc[index_xs]
        # Agregate the cluster xr
        df_list.append(xs_cluster)
        #Remove samples already aggregated
        principal=principal.drop(index=index_xs)

    if (principal.shape[0]>=k) and (principal.shape[0]<=2*k-1):
        df_list.append(principal)

    if principal.shape[0]<k:
       mean_list=[]
       index_principal=principal.index.to_list()
      #Calculate the mean of each cluster and create a list of mean values for each cluster
       for i in range(len(df_list)):
            #Calculate the mean of each cluster
            mean_df=df_list[i].mean()
            #Create a list of mean values for each cluster
            mean_list.append(mean_df)

       for i in range(principal.shape[0]):
            mean_df=pd.DataFrame(mean_list)
            #Find the distance of the cluster mean
            mean_prov=(mean_df-principal.iat[i])**2
            #index of the min distance
            index=mean_prov.idxmin(numeric_only=True)
            df_prov=df_list[index.iloc[0]]
            df_prov[index_principal[i]]=principal.iat[i]
            df_list[index.iloc[0]]=df_prov

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