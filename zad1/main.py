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

print("4b\n")

with open('diabetes.txt', 'r') as file:
    values = np.loadtxt(file)


min_value = np.min(values)
max_value = np.max(values)
scaled_values = (values - min_value) / (max_value - min_value)

a1 = -1
b1 = 1
normalized_values1 = a1 + (b1 - a1) * scaled_values

print(normalized_values1)

print("\n--------------------------------------\n")

a2 = 0
b2 = 1
normalized_values2 = a2 + (b2 - a2) * scaled_values

print(normalized_values2)

print("\n--------------------------------------\n")

a3 = -10
b3 = 10
normalized_values3 = a3 + (b3 - a3) * scaled_values

print(normalized_values3)

print("\n--------------------------------------\n")

print("4c\n")

with open('diabetes.txt', 'r') as file:
    values = np.loadtxt(file)

i = 2
ai = values[:, i]
ai_mean = np.mean(ai)
ai_std = np.std(ai)
ai_standardized = (ai - ai_mean) / ai_std

print(ai_standardized)
