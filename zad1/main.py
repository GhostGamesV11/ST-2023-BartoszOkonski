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
    temp = np.array(diabetes[:, i], dtype='float')
    min = np.min(temp)
    max = np.max(temp)
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
    temp = np.array(diabetes[:, i], dtype='float')
    wholeSystem = np.append(wholeSystem, temp)
    print(f"{diabetesType[i][0]}: std = {np.std(temp)}")
print(f"Whole system: std = {np.std(wholeSystem)}")

print("\n--------------------------------------\n")

print("4a\n")
x1, y1 = np.where(diabetesType == "n")
for i in x1:
    temp = np.array(diabetes[:, i], dtype='float')
    avg = 0
    for j in temp:
        avg += j
    avg = np.round(avg / len(temp), 2)
    print(avg)
    break

dataChoice = np.random.choice([True, False], size=diabetes.shape, p=[0.1, 0.9])
diabetes[dataChoice] = "?"
print(diabetes)



