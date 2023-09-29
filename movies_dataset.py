"""
Manage movies data (extracted from /data/MovieGenre.csv).
"""

import io
import os.path
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

from matplotlib import pyplot as plt
import struct

images_folder = 'data/images/'
test_data_ratio = 7  # 14.3%
validation_data_ratio = 6  # 14.3%
parsed_movies = []  # cache


class Movie:
	imdb_id = 0
	title = ''
	year = 0
	genres = []
	poster_url = ''

	# milix: added size in parameters
	def poster_file_exists(self,size=100) -> bool:
		return os.path.isfile(self.poster_file_path(size))

	def download_poster(self):
		try:
			response = urllib.request.urlopen(self.poster_url)
			data = response.read()
			file = open(self.poster_file_path(), 'wb')
			file.write(bytearray(data))
			file.close()
			return data
		except:
			print('-> error')

	def poster_file_path(self, size=100) -> str:
		return images_folder + str(size) + "/" + self.poster_file_name()

	def poster_file_name(self):
		return str(self.imdb_id) + '.jpg'

	def is_valid(self) -> bool:
		return self.poster_url.startswith('https://') \
				and 1900 <= self.year <= 2018 \
				and len(self.title) > 1 \
				and len(self.genres) > 1

	def to_rgb_pixels(self, poster_size):
		data = open(images_folder + str(poster_size) + '/' + str(self.imdb_id) + '.jpg', "rb").read()
		image = Image.open(io.BytesIO(data))
		rgb_im = image.convert('RGB')
		pixels = []
		for x in range(image.size[0]):
			row = []
			for y in range(image.size[1]):
				r, g, b = rgb_im.getpixel((x, y))
				pixel = [r / 255, g / 255, b / 255]
				row.append(pixel)
			pixels.append(row)
		return pixels, image #new (image)

	# milix
	def read_image_rgb(self, poster_size):
		data = open(images_folder + str(poster_size) + '/' + str(self.imdb_id) + '.jpg', "rb").read()
		try:
			image = Image.open(io.BytesIO(data))
		except:
			return None
		rgb_im = image.convert('RGB')
		return rgb_im

	def get_genres_vector(self, genres):
		if len(genres) == 1:
			has_genre = self.has_genre(genres[0])
			return [int(has_genre), int(not has_genre)]
		else:
			vector = []
			if self.has_any_genre(genres):
				for genre in genres:
					vector.append(int(self.has_genre(genre)))

			return vector

	def short_title(self) -> str:
		max_size = 20
		return (self.title[:max_size] + '..') if len(self.title) > max_size else self.title

	def is_test_data(self) -> bool:
		return self.imdb_id % test_data_ratio == 0

	def has_any_genre(self, genres) -> bool:
		return len(set(self.genres).intersection(genres)) > 0

	def has_genre(self, genre) -> bool:
		return genre in self.genres

	def __str__(self):
		return self.short_title() + ' (' + str(self.year) + ')'


def download_posters(min_year=0):
	for movie in list_movies():
		print(str(movie))
		if movie.year >= min_year:
			if not movie.poster_file_exists():
				movie.download_poster()
				if movie.poster_file_exists():
					print('-> downloaded')
				else:
					print('-> could not download')
			else:
				print('-> already downloaded')
		else:
			print('-> skip (too old)')


def load_genre_data(min_year, max_year, genres, ratio, data_type, verbose=True):
	xs = []
	ys = []
	for year in reversed(range(min_year, max_year + 1)):
		if verbose:
			print('loading movies', data_type, 'data for', year, '...')
		xs_year, ys_year = _load_genre_data_per_year(year, genres, ratio, data_type)
		_add_to(xs_year, xs)
		_add_to(ys_year, ys)
		if verbose:
			print('->', len(xs_year))
	return np.concatenate(xs), np.concatenate(ys)


def _load_genre_data_per_year(year, genres, poster_ratio, data_type):
	xs = []
	ys = []

	count = 1
	for movie in list_movies(year, genres):
		if movie.poster_file_exists():
			if (data_type == 'train' and not movie.is_test_data() and count % validation_data_ratio != 0) \
					or (data_type == 'validation' and not movie.is_test_data() and count % validation_data_ratio == 0) \
					or (data_type == 'test' and movie.is_test_data()):
				x, img = movie.to_rgb_pixels(poster_ratio)  #new(img)
				y = movie.get_genres_vector(genres)
				xs.append(x)
				ys.append(y)
			count += 1

	xs = np.array(xs, dtype='float32')
	ys = np.array(ys, dtype='uint8')
	return xs, ys


def _add_to(array1d, array2d):
	if len(array1d) > 0:
		array2d.append(array1d)



'''
milix_create_binfiles
create fx (images) anf fy (genres=labels)

fx format
---------
nitems:  int32
height:  int32
width:   int32
image(1): imdb_id[int32], rrrrggggbbbb
image(2): imdb_id[int32], rrrrggggbbbb
..., 
image(nitems)

'''
def milix_create_binfiles(xfilename, yfilename, ratio, genres):
	min_year = 1977
	max_year = 2017
	totcount = 0  # nitems (will be updated at the end) 
	fx = open(xfilename, "wb")
	fy = open(yfilename, "wb")
	fx.write((totcount).to_bytes(4, byteorder='big', signed=False))
	fy.write((totcount).to_bytes(4, byteorder='big', signed=False))
	for year in reversed(range(min_year, max_year+1)):
		print('adding movie data for ', year, '...')
		count = 0
		movies = list_movies(year, genres)
		for movie in movies:
			if movie.poster_file_exists(ratio):
				img = movie.read_image_rgb(ratio)
				if not img:
					continue
				gen = movie.get_genres_vector(genres)
				if (totcount==0):
					# first image, write height and width to the file
					fx.write((img.size[1]).to_bytes(4, byteorder='big', signed=False)) # height
					fx.write((img.size[0]).to_bytes(4, byteorder='big', signed=False)) # width
					# write genres size
					fy.write((len(gen)).to_bytes(4, byteorder='big', signed=False))
					 
				# write imdb_id
				fx.write((movie.imdb_id).to_bytes(4, byteorder='big', signed=False))
				# write pixel data (first r then g and then b)
				red = []
				green = []
				blue = []
				for y in range(img.size[1]):
					for x in range(img.size[0]):
						r, g, b = img.getpixel((x, y))
						red.append(r)
						green.append(g)
						blue.append(b)

				fx.write(bytearray(red))
				fx.write(bytearray(green))
				fx.write(bytearray(blue))

				'''
				pixels = np.array(blue).reshape((img.size[1], img.size[0]))
				plt.imshow(pixels, cmap='gray')
				plt.show()
				'''
				# write genres
				#print(movie.imdb_id, gen)
				fy.write(bytearray(gen))
				
				count += 1
				totcount += 1
				
		if count > 0: 
			print(f"Found {count} movies in year {year}.")
			
	fx.seek(0)
	fx.write((totcount).to_bytes(4, byteorder='big', signed=False))
	fx.close()
	fy.seek(0)
	fy.write((totcount).to_bytes(4, byteorder='big', signed=False))
	fy.close()


def milix_load_posters(fname):
	fx = open(fname, "rb")
	n = struct.unpack('>I', fx.read(4))[0]
	h = struct.unpack('>I', fx.read(4))[0]
	w = struct.unpack('>I', fx.read(4))[0]
	print(f"Going to read {n} posters of size {h}x{w}")
	X = []
	ids = []
	for i in range(n):
		imdb_id = struct.unpack('>I', fx.read(4))[0]
		ids.append(imdb_id)
		r = np.array(bytearray(fx.read(h*w))).reshape((h,w))
		g = np.array(bytearray(fx.read(h*w))).reshape((h,w))
		b = np.array(bytearray(fx.read(h*w))).reshape((h,w))
		image = np.dstack((r,g,b))
		# bottleneck in performance here
		'''
		image = np.zeros((h,w,3))
		for y in range(h):
			for x in range(w):
				image[y,x,0] = r[y,x];
				image[y,x,1] = g[y,x];
				image[y,x,2] = b[y,x];
		'''
		X.append(image/255.0)
	
	fx.close()	
	X = np.array(X, dtype='float32')
	ids = np.array(ids, dtype='uint32')
	return X, ids


def milix_load_genres(fname):
	fy = open(fname, "rb")
	n = struct.unpack('>I', fy.read(4))[0]
	sz = struct.unpack('>I', fy.read(4))[0]
	y = []
	for i in range(n):
		g = []
		for j in range(sz): g.append(int.from_bytes(fy.read(1), 'big'))
		y.append(g)
		
	fy.close()
	y = np.array(y, dtype='uint8')
	return y


def list_movies(year=None, genres=None):
	if len(parsed_movies) == 0:
		data = pd.read_csv('data/MovieGenre.csv', encoding='ISO-8859-1')
		for index, row in data.iterrows():
			movie = _parse_movie_row(row)
			if movie.is_valid():
				parsed_movies.append(movie)

		parsed_movies.sort(key=lambda m: m.imdb_id)

	result = parsed_movies

	if year is not None:
		result = [movie for movie in result if movie.year == year]

	if genres is not None:
		result = [movie for movie in result if movie.has_any_genre(genres)]

	return result


def _parse_movie_row(row) -> Movie:
	movie = Movie()
	movie.imdb_id = int(row['imdbId'])
	movie.title = row['Title'][:-7]
	year = row['Title'][-5:-1]
	if year.isdigit() and len(year) == 4:
		movie.year = int(row['Title'][-5:-1])

	url = str(row['Poster'])
	if len(url) > 0:
		movie.poster_url = url.replace('"', '')

	genre_str = str(row['Genre'])
	if len(genre_str) > 0:
		movie.genres = genre_str.split('|')

	return movie


def search_movie(imdb_id=None, title=None) -> Movie:
	movies = list_movies()
	for movie in movies:
		if imdb_id is not None and movie.imdb_id == imdb_id:
			return movie
		if title is not None and movie.title == title:
			return movie


def list_genres(number):
	if number == 3:
		return ['Comedy', 'Drama', 'Action']
	if number == 7:
		return list_genres(3) + ['Animation', 'Romance', 'Adventure', 'Horror']
	if number == 14:
		return list_genres(7) + ['Sci-Fi', 'Crime', 'Mystery', 'Thriller', 'War', 'Family', 'Western']
