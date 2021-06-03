import numpy as np
import pandas as pd
import xlrd
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_excel(
    'Data_Sets.xlsx', sheet_name="Data Set 2", header=None).values.tolist()


train_list = []
test_list = []
for i in range(len(data)):
    if(i == 0):
        train_list.append(data[i])
    elif(i > len(data)-21):
        test_list.append(data[i])
    elif((i+1) % 5 == 0):
        # print(i)
        test_list.append(data[i])
    else:
        train_list.append(data[i])


df = pd.DataFrame(test_list, columns=['X', 'Y'])
df.to_excel('test.xlsx', index=False)
test_data = pd.read_excel('test.xlsx').values.tolist()

# df1 = pd.DataFrame(train_list, columns=['X', 'Y'])
# df1.to_excel('train.xlsx', index=False)
# train_list = pd.read_excel('train.xlsx').values.tolist()

# # print(test_list)


def fuzzy(u, c, m, length, dataset):  # to calculate centroids of the clusters
    v = []
    for i in range(c):
        x_sum1 = 0
        x_sum2 = 0
        y_sum1 = 0
        y_sum2 = 0
        for j in range(len(dataset)):
            x_sum1 = x_sum1 + (u[i][j]**2)*dataset[j][0]
            x_sum2 = x_sum2 + u[i][j]**2
            y_sum1 = y_sum1 + (u[i][j]**2)*dataset[j][1]
            y_sum2 = y_sum2 + u[i][j]**2
        x = x_sum1/x_sum2
        y = y_sum1/y_sum2
        v.append([x, y])
        A_len = len(dataset[0])
    A = np.identity(A_len, dtype=float)
    d = []
    for i in range(len(v)):
        temp_d = []
        for j in range(len(dataset)):
            temp1 = np.transpose(np.array(dataset[j])-np.array(v[i]))
            temp2 = np.dot(temp1, A)
            temp3 = np.dot(temp2, (np.array(dataset[j])-np.array(v[i])))
            # print(np.shape(temp3))
            temp_d.append(temp3)
        d.append(temp_d)

    new_u = []
    for i in range(len(d)):
        temp = []
        for k in range(len(d[0])):
            sum = 0
            if(d[i][k] == 0):
                u_sum = 0
                temp.append(u_sum)
                break
            for j in range(len(d)):
                sum = sum + (d[i][k]/d[j][k])**2
            u_sum = 1/sum
            temp.append(u_sum)
        new_u.append(temp)
    error = np.linalg.norm(np.array(new_u)-np.array(u))
    return error, new_u, v
    # if(error > 0.001):
    #     u = new_u
    #     fuzzy(u, c, 2, len(train_list), train_list)
    # else:
    #     print(new_u)


def jFunction(u, dataset, v):
    A_len = len(dataset[0])
    A = np.identity(A_len, dtype=float)
    final_sum = 0
    for i in range(len(v)):
        sum = 0
        for k in range(len(dataset)):
            temp1 = np.transpose(np.array(dataset[k])-np.array(v[i]))
            temp2 = np.dot(temp1, A)
            temp3 = np.dot(temp2, (np.array(dataset[k])-np.array(v[i])))
            sum = sum + u[i][k]**2*temp3
        final_sum = final_sum + sum
    return final_sum


def maxi_col(mat, rows, cols):
    temp = []
    for i in range(cols):
        maxi = mat[0][i]
        for j in range(rows):
            if(mat[j][i] > maxi):
                maxi = mat[j][i]
        temp.append(maxi)
    return temp


c = 2
j = []
iteration_count = []
highest_points = []
while(c < 11):
    u = np.random.randint(2, size=(c, len(train_list))).tolist()
    error, new_u, v = fuzzy(u, c, 2, len(train_list), train_list)
    count = 0
    while(error > 0.001):
        u = new_u
        error, new_u, v = fuzzy(u, c, 2, len(train_list), train_list)
        count = count+1
    c = c+1

    # print(v)

    rows = len(new_u)
    cols = len(new_u[0])
    highest_points.append(maxi_col(new_u, rows, cols))
    # print(c-1)
    # print(len(highest_points[c-3]))
    # print(len(new_u))
    # print('/n')
    value = jFunction(new_u, train_list, v)
    j.append(value)
    iteration_count.append(count)

# print(iteration_count)
# print(j)

r = []
for i in range(len(j)):
    if(i == 0):
        continue
    elif(i == len(j)-1):
        continue
    else:
        r_temp = abs((j[i]-j[i+1])/(j[i-1]-j[i]))
        r.append(r_temp)
# print(r)
# print(r.index(min(r))+3)
c = r.index(min(r))+3
print(c)
u = np.random.randint(2, size=(c, len(train_list))).tolist()
error, new_u, v = fuzzy(u, c, 2, len(train_list), train_list)
while(error > 0.001):
    u = new_u
    error, new_u, v = fuzzy(u, c, 2, len(train_list), train_list)
# print(new_u)
print(v)

c = [2, 3, 4, 5, 6, 7, 8, 9, 10]
fig, plot1 = plt.subplots()
plot1.plot(c, j, label="line 1", color="red")
plot1.set_xlabel("c", fontsize=14)
plot1.set_ylabel("J value", color="red", fontsize=14)
plot2 = plot1.twinx()
plot2.plot(c, iteration_count, label="line 2", color="orange")
plot2.set_ylabel("No of iterations", color="blue", fontsize=14)
# plt.show()
plt.savefig("figure1.png")
plt.clf()

c_value_points = []

for i in range(len(highest_points[0])):
    maxi = highest_points[0][i]
    for j in range(len(highest_points)):
        if(highest_points[j][i] > maxi):
            maxi = highest_points[j][i]
            index = j
    c_value_points.append(index)


final_c_value_points = [x + 2 for x in c_value_points]
# print(final_c_value_points)


# df1 = pd.DataFrame(train_list, columns=['X', 'Y'])
# df1.to_excel('train.xlsx', index=False)
def putInExcel(dataset, final_c_value_points):
    for i in range(len(dataset)):
        dataset[i].append(final_c_value_points[i])
    return dataset


cluster_dataset = putInExcel(train_list, final_c_value_points)
df1 = pd.DataFrame(cluster_dataset, columns=['X', 'Y', 'cluster'])
df1.to_excel('C.xlsx', index=False)


with open('final_output.txt', 'w') as f:
    f.write(json.dumps(v))


def test(test_list):
    with open('final_output.txt', 'r') as f:
        v = json.loads(f.read())

    # test_data = pd.read_excel('test.xlsx').values.tolist()
    # print(len(test_data))

    def fuzzy_test(c, m, v, length, dataset):
        A_len = len(dataset[0])
        A = np.identity(A_len, dtype=float)
        d = []
        for i in range(len(v)):
            temp_d = []
            for j in range(len(dataset)):
                temp1 = np.transpose(np.array(dataset[j])-np.array(v[i]))
                temp2 = np.dot(temp1, A)
                temp3 = np.dot(temp2, (np.array(dataset[j])-np.array(v[i])))
                # print(np.shape(temp3))
                temp_d.append(temp3)
            d.append(temp_d)

        new_u = []
        for i in range(len(d)):
            temp = []
            for k in range(len(d[0])):
                sum = 0
                if(d[i][k] == 0):
                    u_sum = 0
                    temp.append(u_sum)
                    break
                for j in range(len(d)):
                    sum = sum + (d[i][k]/d[j][k])**2
                u_sum = 1/sum
                temp.append(u_sum)
            new_u.append(temp)
            # print(temp)
            # print("\n")
        return new_u

    def maxi_coloum(mat, rows, cols, index):
        temp = []
        for i in range(cols):
            maxi = mat[0][i]
            for j in range(rows):
                if(mat[j][i] > maxi):
                    maxi = mat[j][i]
                    index = j
            temp.append(index)

        return temp

    u = fuzzy_test(len(v), 2, v, len(test_list), test_list)
    # print(u)
    point_cluster = []
    rows = len(u)
    cols = len(u[0])
    point_cluster.extend(maxi_coloum(u, rows, cols, 0))
    # print(len(point_cluster))

    for i in range(len(test_list)):
        test_list[i].append(point_cluster[i])

    test_list = np.array(test_list)
    point_cluster = np.array(point_cluster)
    v = np.array(v)

    colors = cm.rainbow(np.linspace(0, 1, v.shape[0]))
    plt.scatter(test_list[:, 0], test_list[:, 1],
                c=colors[test_list[:, 2].astype(int)], s=4)
    plt.scatter(v[:, 0], v[:, 1], c='black', s=50)

    # plt.show()
    plt.savefig("figure2.png")


test(test_list)
