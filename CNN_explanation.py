class ActorClassifierCNN(nn.Module):
    def __init__(self):
        super(ActorClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)  # Ajith or Vijay
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))  # -> (128, 16, 16)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

  A color image comes with 3 channels (RGB). Example:

Say the image is of size 128×128×3 (height × width × color channels).

Now, the conv1 layer learns 32 different filters (each filter tries to capture a different feature like edges, corners, textures etc.).

Each filter scans across the input image and creates a new "feature map".

So after this layer,
the image shape becomes:

128×128×32
