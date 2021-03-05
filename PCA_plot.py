#! usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

def PCA_plot(X, Y):
    pca = PCA(n_components=3, svd_solver='full').fit(X)
    print(pca.explained_variance_ratio_)

    X_pca = pca.transform(X)
    print(X_pca.shape)

    # 3d
    fig = plt.figure(figsize=(10,8))
    # 
    ax = fig.gca(projection='3d')
    scatter3D = ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=Y, cmap='jet', edgecolors='k', vmin=0.1, vmax=Y.max())
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    # cbar = plt.colorbar(scatter3D)
    # cbar.set_label('A2_pore/wt ratio', rotation=270, va='bottom')
    # for i in range(X_pca.shape[0]):
    # ax.text(X_pca[i,0], X_pca[i,1], X_pca[i,2], str(i))
    plt.show()

    # 2D
    fig = plt.figure(figsize=(10,8))
    scatter2D = plt.scatter(X_pca[:,0], X_pca[:,1], c=Y, cmap='jet', edgecolors='k')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # cbar = plt.colorbar(scatter2D)
    # cbar.set_label('A2_pore/wt ratio', rotation=270, va='bottom')
    plt.show()
    # annotate each point

    plt.show()

    pca_contribution = pd.DataFrame(pca.components_,columns=X.columns,index = ['PC-1','PC-2','PC-3'])
    print(pca_contribution)

    return 0