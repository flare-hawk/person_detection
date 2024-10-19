import scipy.io as sio 
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt 
matplotlib.use('Agg')


# Load the YOLO v11 x model 
model=YOLO("./model/yolo11x.pt")

# Predictions # Class 0 is the label
results=model.predict("./data/frames", device="cuda:0", classes=[0])

# Laod the Ground Truth
orig_data=sio.loadmat("./data/mall_gt.mat")
ground_truth=np.squeeze(orig_data['count'])

# Compare the Results 
# Extract counts from YOLO  
yolo_counts = []
for idx, result in enumerate(results):
    yolo_counts.append(len(result.boxes))


print(f"YOLO Busiest Frame: {np.argmax(yolo_counts)+1}")
print(f"YOLO Busiest Frame: {np.argmax(ground_truth)+1}")

# Percent
mse = np.mean((np.array(yolo_counts)-ground_truth)**2)
print(f"Mean Squared Error: {mse}")

# Average Predicted Counts as % of True Counts 
ap_percent = np.mean((np.array(yolo_counts)/ground_truth))
print(f"Average % Prediction: {ap_percent}")

# Check Accuracy 
checked_counts=[0 for _ in range(2000)]
for i in range(2000):
    for box in results[0].boxes:
        bbox = np.array(box.xyxy.cpu())[0]
        heads=orig_data['frame'][0][i][0][0][0]
        for head in heads:
            if bbox[0]<=head[0]<=bbox[2] and bbox[1]<=head[1]<=bbox[3]:
                checked_counts[i]+=1
                break

# Percent
mse_checked = np.mean((np.array(checked_counts)-ground_truth)**2)
print(f"Mean Squared Error: {mse}")

# Average Predicted Counts as % of True Counts 
ap_percent_checked = np.mean((np.array(checked_counts)/ground_truth))
print(f"Average % Prediction: {ap_percent_checked}")    

# A scatter plot
plt.scatter(ground_truth, yolo_counts, color='blue', label='Predicted vs True')
plt.plot([0, max(ground_truth)], [0, max(ground_truth)], color='red', linestyle='--', label='Perfect Prediction')

# Labeling the plot
plt.title('Scatter Plot of True Counts vs Predicted Counts')
plt.xlabel('True Counts')
plt.ylabel('Predicted Counts')
plt.legend()

plt.savefig('scatter_plot.png')
plt.close()



# A scatter plot
plt.scatter(ground_truth, checked_counts, color='blue', label='Predicted vs True')
plt.plot([0, max(ground_truth)], [0, max(ground_truth)], color='red', linestyle='--', label='Perfect Prediction')

# Labeling the plot
plt.title('Scatter Plot of True Counts vs Predicted Counts')
plt.xlabel('True Counts')
plt.ylabel('Predicted Counts')
plt.legend()

plt.savefig('scatter_plot_checked.png')
plt.close()


