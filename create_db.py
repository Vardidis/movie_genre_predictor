import movies_dataset as movies

import time
from matplotlib import pyplot as plt

min_year = 1977
max_year = 2017
epochs = 50
ngenres = 7
genres = movies.list_genres(ngenres)

print('Creating file...')

t1 = time.perf_counter()
for ratio in [30]:
    # we load the data once for each ratio, so we can use it for multiple versions, epochs, etc.
    xfile = "data/posters"+str(ratio)+".bin"
    yfile = "data/posters"+str(ratio)+"_"+str(ngenres)+"genres.bin"
    movies.milix_create_binfiles(xfile, yfile, ratio, genres)
t2 = time.perf_counter()
print(f"Done in {t2-t1:0.4f} seconds.")

# load and show images
ratio = 30
xfile = "data/posters"+str(ratio)+".bin"
yfile = "data/posters"+str(ratio)+"_"+str(ngenres)+"genres.bin"
t1 = time.perf_counter()
X, ids = movies.milix_load_posters(xfile)
y = movies.milix_load_genres(yfile)
t2 = time.perf_counter()
print(f"Done in {t2-t1:0.4f} seconds.")
print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(ids[2])
print(y[2])
pixels = X[2]
plt.imshow(pixels) #, cmap='gray')
plt.show()


print('Done.')
