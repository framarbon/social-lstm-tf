
import csv
import numpy as np

f = open('pixel_pos_interpolate.csv')
f2 = open('pixel_pos_interpolate_updated.csv', 'wb')
reader = csv.reader(f)
writer = csv.writer(f2)
data = []
for row in reader:
        data.append(map(float,row))

datan = np.asarray(data)
datan[1] += 1

for row in datan:
    writer.writerow(row)


