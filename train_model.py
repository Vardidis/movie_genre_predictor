import movies_dataset as movies
import movies_milix_model

import os
import time
from sklearn.model_selection import train_test_split

min_year = 1977
max_year = 2017
epochs = 60
ratio = 30
ngenres = 3
genres = movies.list_genres(ngenres)

#vardis
if not os.path.isdir('data/posters'):
    os.makedirs('data/posters')
    open(str(ratio)+".bin", 'w').close()
    open(str(ratio)+"_"+str(ngenres)+"genres.bin", 'w').close()

xfile = "data/posters"+str(ratio)+".bin"
yfile = "data/posters"+str(ratio)+"_"+str(ngenres)+"genres.bin"

#vardis
print('Creating binfiles...')
if not (os.path.isfile("data/posters"+str(ratio)+".bin") and os.path.isfile("data/posters"+str(ratio)+"_"+str(ngenres)+"genres.bin")):
    movies.milix_create_binfiles(xfile, yfile, ratio, genres)

#milix
print('Loading datasets...')
t1 = time.perf_counter()
X, ids = movies.milix_load_posters(xfile)
y = movies.milix_load_genres(yfile)
t2 = time.perf_counter()
print(f"Done in {t2-t1:0.4f} seconds.")

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=49)
X_train = X 
y_train = y
X_test = X
y_test = y

# training version (1, 2 or 3)
version = 1
movies_milix_model.build(version, min_year, max_year, genres, ratio, epochs,
                         x_train=X_train,
                         y_train=y_train,
                         x_validation=X_test,
                         y_validation=y_test)

print('Done.')
