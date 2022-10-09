#
# for i in range(10):
#     try:
#         print(i)
#         print(5/0)
#     except Exception as e:
#         print("--")


# from scipy import spatial
# cosine_similarity = lambda vec1, vec2: 1 - spatial.distance.cosine(vec1, vec2)
#
# box = cosine_similarity([1,2,9], [1,2,3])
# print(box)


# box = [[1,100], [2,101], [3,50]]
# print(sorted(box, key=lambda item:item[1], reverse=True))


# box = {'neg': 1, 'neu': 5, 'pos': 9, 'compound': 7163}
# srt = sorted(box.items(), key=lambda item:item[1], reverse=True)
# print(srt)

# dict = [[1,10], [50,11], [6,4], [9,3]]
# box = sorted(dict, key=lambda x:x[1])
# print(box)

# funct = lambda x, y: x*y
# box = funct(3,4)
# print(box)

# import numpy as np
# mat = np.array([23,445,645,34,35,1])
# print(mat)
# print(mat.argsort())
# print(mat.argsort()[-2:])


# list = [1,2,3,4,5,6]
# print(list[1:4])


# import numpy as np
# input_array = np.array([[1, 2, 3],
#                           [4, 5, 6],
#                           [6, 0, 1]])
# print(input_array)
#
# from keras.models import Sequential
# from keras.layers import Embedding
#
# model = Sequential()
# model.add(Embedding(input_dim=9, output_dim=10, input_length=3))
#
# model.compile(loss='mse', optimizer='rmsprop')
# output_array = model.predict(input_array)
#
# print(output_array)
#
# import numpy as np
# y = np.zeros(3+1)
# print(y)
