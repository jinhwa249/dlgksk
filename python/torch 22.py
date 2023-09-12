class LayerActivations:
    features = []
    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        
    def hook_fn(self, model, input, output):
        self.features = output.detach().numpy()
        
    def remove(self):
        self.hook.remove()
        
img = cv2.imread(r"c:\Users\admin\Downloads\cat.jpg")
plt.imshow(img)
img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
img.TOTensor()(img).nusqueeze(0)
print(img.shape)

result = LayerActivations(model.features, 0)

model(img)
acrivations = result.features

fig, axes = plt.subplots(4, 4)
fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0, righr=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for row in range(4):
    for column in range(4):
        axis = axes[row][column]
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        axis.imshow(activations[0][row*10+column])
plt.show()
