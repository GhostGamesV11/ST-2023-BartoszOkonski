import numpy as np

diabetes = np.loadtxt("diabetes.txt", dtype="str")
diabetesType = np.loadtxt("diabetes-type.txt", dtype="str")
infoData = np.loadtxt("_info-data-discrete.txt", dtype="str")

print("\n--------------------------------------\n")

print("3b\n")
x, y = np.where(infoData == "diabetes")
print(f"size of decision classes = {infoData[int(x)][2]}")

print("\n--------------------------------------\n")

print("3c\n")
x1, y1 = np.where(diabetesType == "n")
for i in x1:
    tym = np.array(diabetes[:, i], dtype='float')
    max = np.max(tym)
    min = np.min(tym)
    print(f"{diabetesType[i][0]}: max = {max}, min = {min}")

print("\n--------------------------------------\n")


print("3d\n")
for i in range(len(diabetes[0]) - 1):
    print(f"{diabetesType[i][0]}: uniqueAttributes = {len(np.unique(diabetes[:, i]))}")

print("\n--------------------------------------\n")

print("3e\n")
for i in range(len(diabetes[0]) - 1):
    print(f"{diabetesType[i][0]}: uniqueAttributes = {np.unique(diabetes[:, i])}")


print("\n--------------------------------------\n")


print("3f\n")
x1, y1 = np.where(diabetesType == "n")
wholeSystem = np.array([])
for i in x1:
    tym = np.array(diabetes[:, i], dtype='float')
    wholeSystem = np.append(wholeSystem, tym)
    print(f"{diabetesType[i][0]}: std = {np.std(tym)}")
print(f"Whole system: std = {np.std(wholeSystem)}")
