import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
#importing required libereries
sns.set_theme(style='whitegrid')
#%matplotlib inline
# df=pd.read_csv("Clean_Dataset.csv")    #loading data present in the same folder
# df.head()
# df.info()
# df.describe()
# df.columns
# df.isnull().sum()
#data checking 


# Load dataset
df = pd.read_csv("Clean_Dataset.csv")

# Label encoding for categorical columns
categorical_cols = ['source_city', 'destination_city', 'class', 'departure_time']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for possible inverse_transform later

# Select features and target
features = ['source_city', 'destination_city', 'class', 'duration', 'days_left', 'departure_time']
X = df[features]
y = df['price']

# Convert to numpy arrays
x_array = np.array(X)
y_array = np.array(y)

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_array)


def calc_cost(x,y,w,b):
    # Calculate the cost function
    m=x.shape[0]
    total_cost=0
    cost=0
    for i in range(m):
        fx=0
        fx=np.dot(w,x[i])+b
        cost+=(fx-y[i])**2
    total_cost=cost/(2*m)
    return total_cost
i_w=[0.1,0.1,0.1,0.1,0.1,0.1]
init_w=np.array(i_w)
init_b=0
x=calc_cost(x_array,y_array,init_w,init_b)
# print("cost :",x)
# checking cost for some random values of w and b 
def calc_gradient(x, y, w, b):
    m = x.shape[0]
    fx = np.dot(x, w) + b        
    error = fx - y               
    dw = np.dot(x.T, error) / m 
    db = np.sum(error) / m     
    return dw, db
i_w=[0.1,0.1,0.1,0.1,0.1,0.1]
init_w=np.array(i_w)
init_b=0
grd_w, grd_b=calc_gradient(x_array,y_array,init_w,init_b)
# print(grd_w,"\n",grd_b)

def gradient_desent(x,y,w,b,calc_gradient,calc_cost,alpha,iterations):
    w_in=w
    b_in=b
    cost_his=[]
    for i in range(iterations):
        dw, db=calc_gradient(x,y,w_in,b_in)
        w_in=w_in-(alpha*dw)
        b_in=b_in-(alpha*db)
        # if i % 10 == 0:
        #     print(f"dw: {dw}, db: {db}")
        cost=calc_cost(x,y,w_in,b_in)
        cost_his.append(cost)
        # if i % max(1, iterations // 10) == 0:
            # print(f"Iteration {i}: Cost {cost:.4f}")
    return w_in,b_in,cost_his
i_w=[0,0,0,0,0,0]
init_w=np.array(i_w)
init_b=0
alpha=0.15
itera=40


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_array)

fin_w, fin_b, cost_history=gradient_desent(x_scaled,y_array,init_w,init_b,calc_gradient,calc_cost,alpha,itera)
# print(fin_w,"\n")

# print("b value is",fin_b)
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Time")
plt.grid(True)
plt.show()

#[   120.7163151     157.32307012 -21031.78494348   1644.45171641
#-1746.91093002     61.35735584]  20839.42069548274
m=len(fin_w)
x_in=[]
for i in range(m):
    x_temp=float(input("Enetr values :"))   # input format ['source_city', 'destination_city', 'class', 'duration', 'days_left', 'departure_time']
    x_in.append(x_temp)
x_in=np.array(x_in).reshape(1,-1)
x_scaled=scaler.transform(x_in)
y_x=np.dot(fin_w,x_scaled[0])+fin_b
print(f"prediction :{y_x:.2f}")
