from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

A = imread('cane.png')
# A è una matrice  M x N x 3, essendo un'immagine RGB
# A(:,:,1) Red A(:,:,2) Blue A(:,:,3) Green
# su una scala tra 0 e 1
print(A.shape)


X = np.mean(A,-1); # media lungo l'ultimo asse, cioè 2
img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()


# If full_matrices=True (default), u and vT have the shapes (M, M) and (N, N), respectively.
# Otherwise, the shapes are (M, K) and (K, N), respectively, where K = min(M, N).
U, S, VT = np.linalg.svd(X,full_matrices=False)
print(S[100:105])
S = np.diag(S)

j=0
energy = 0
best_k = 0

# Determinare il valore minimo di k per cui si conserva l’80% dell’energia totale. 
for k in range(1, S.shape[0]):
	sum1 = 0
	sum2 = 0
	Xapprox = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
	
	# somma dei primi k valori singolari
	for i in range(0,k):
		sum1 += S[i][i]

	# somma di tutti i valori singolari
	for i in range(0, S.shape[0]):
		sum2 += S[i][i]

	energy = sum1 / sum2

	if energy >= 0.8:
		best_k = k
		break

for k in (5,20,100,best_k):
	sum1 = 0
	sum2 = 0
	Xapprox = U[:,:k] @ S[0:k,:k] @ VT[:k,:]
	
	for i in range(0,k):
		sum1 += S[i][i]

	for i in range(0, S.shape[0]):
		sum2 += S[i][i]

	energy = sum1 / sum2

	plt.figure(j+1)
	j +=1
	img = plt.imshow(Xapprox)
	img.set_cmap('gray')
	plt.axis('off')
	plt.title(f'r = {k} - energy = {round(energy,2)}')

plt.show()
