import sys
import numpy as np
from .utils import read_xyz_file, read_coord_file, shift_xyz_coord, shift_coord_file, write_xyz_file, write_coord_file

def find_min_max(xyz, coord):
	"""
	Finds maximum and minium z value of junction for symmetric shift
	"""
	max_z = -1E12
	min_z = +1E12
	if(xyz == True):
		for i in range(0,coord.shape[1]):
			tmp = float(coord[3,i])
			if(tmp > max_z):
				max_z = tmp 
			if(tmp < min_z):
				min_z = tmp 
		return min_z, max_z
	else:
		for i in range(0,coord.shape[1]):
			tmp = float(coord[2,i])
			if(tmp > max_z):
				max_z = tmp 
			if(tmp < min_z):
				min_z = tmp 
		return min_z, max_z



		


if __name__ == '__main__':
	"""
	Usage. There are two different modes: Center mode and shift mode. Both take *.xyz files or turbomole files as input. Output has same format. Shift mode just implemented for *.xyz files
	Center Mode:
		shift_coord_fily.py INPUT center Output
	Shift Mode:
		shift_coord_file.py INPUT x_shift y_shift z_shift Output
	"""

	filename_coord = sys.argv[1]
	#check if *.xyz or turbomole format
	xyz = filename_coord.endswith('.xyz')
	if(xyz==True):
		coord = read_xyz_file(filename_coord)
	else:
		coord = read_coord_file(filename_coord)

	#center mode. Centers junction for image charge correction
	if(len(sys.argv)==4 and sys.argv[2] == "center"):
		output = sys.argv[3]
		min_z, max_z = find_min_max(xyz, coord)
		shift = -(min_z+max_z)/2
		if(xyz==True):
			coord_xyz = shift_xyz_coord(coord, 0, 0, shift)
			write_xyz_file(output, coord_xyz)
		else:
			coord = shift_coord_file(coord, 0, 0, shift)
			write_coord_file(output, coord)
		exit(0)
	

	if xyz == True:
		try:
			x_shift = float(sys.argv[2])
			y_shift = float(sys.argv[3])
			z_shift = float(sys.argv[4])
		except ValueError:
			print("Check your arguments. Expected floats")
		output_xyz = sys.argv[5]

		coord_xyz = read_xyz_file(sys.argv[1])
		coord_xyz = shift_xyz_coord(coord_xyz, x_shift, y_shift, z_shift)
		write_xyz_file(output_xyz, coord_xyz)

	else:
		try:
			x_shift = float(sys.argv[2])
			y_shift = float(sys.argv[3])
			z_shift = float(sys.argv[4])
		except ValueError:
			print("Check your arguments. Expected floats")
		output_coord = sys.argv[5]

		coord = read_coord_file(sys.argv[1])
		coord = shift_coord_file(coord, x_shift, y_shift, z_shift)
		write_coord_file(output_coord, coord)