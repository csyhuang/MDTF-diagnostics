import os.path
import sys


def get_parameters_in(lon1, lon2, lat1, lat2, sigma,  imindx1, imindx2,  composite, im1,
                      im2, season,  composite24, regression, correlation,  undef, prefix):
    # read in all  parameter data
    undef = float( 1.1E+20)
    file_path = os.path.join(prefix,"../shared","parameters.txt")

    if os.path.exists(file_path):
        file = open(file_path, 'r')
        line = file.readline()
        line = line.strip()
        column = line.split()
        lon1 = float(column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        lon2 = float(column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        lat1 = float(column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        lat2 = float(column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        sigma = float(column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        imindx1 = int( column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        imindx2 = int( column[2])
        if imindx2 < imindx1:
            imindx2 = imindx2 + 12

        line = file.readline()
        line = line.strip()
        column = line.split()
        composite = int( column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        im1 = int(column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        im2 = int(column[2])
        if im2 < im1:
            im2 = im2 + 12

        line = file.readline()
        line = line.strip()
        column = line.split()
        season = column[2]

#  composite evolution 24 month switches
        line = file.readline()
        line = line.strip()
        column = line.split()
        composite24 = int(column[2])

# regression /correlation
        line = file.readline()
        line = line.strip()
        column = line.split()
        regression = int(column[2])

        line = file.readline()
        line = line.strip()
        column = line.split()
        correlation = int(column[2])

        file.close()

    else:
        print(" missing file: ", file_path)
        print(" exiting get_parameters_in.py ")
        sys.exit(1)
    return (lon1, lon2, lat1, lat2, sigma, imindx1, imindx2, composite,
            im1, im2, season,  composite24, regression, correlation,  undef)