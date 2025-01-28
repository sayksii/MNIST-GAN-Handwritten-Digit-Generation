import torchvision
import torch

# 準備生成器
# 生成器類，繼承 torch.nn.Module
class Generator(torch.nn.Module):

    # 初始化定義模型結構
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        # nn.Sequential 為之前 FCNN 定義多層網絡的改良寫法，可以更簡潔地定義多層網絡
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),  # 全連接層，輸入維度為 input_dim，輸出維度為64
            torch.nn.LeakyReLU(inplace=True),  # 激活函數 ReLU，inplace=True 表示直接對輸入張量進行修改，節省內存
            torch.nn.Linear(64, 128),  # 全連接層，輸入維度為64，輸出維度為128
            torch.nn.LeakyReLU(inplace=True),  # 激活函數 ReLU
            torch.nn.Dropout(p=0.3),  # 丟棄率為 30%
            torch.nn.Linear(128, 256),  # 全連接層，輸入維度為128，輸出維度為256
            torch.nn.LeakyReLU(inplace=True),  # 激活函數 ReLU
            torch.nn.Linear(256, 512),  # 全連接層，輸入維度為256，輸出維度為512
            torch.nn.LeakyReLU(inplace=True),  # 激活函數 ReLU
            torch.nn.Dropout(p=0.3),  # 丟棄率為 30%
            torch.nn.Linear(512, 1024),  # 全連接層，輸入維度為512，輸出維度為1024
            torch.nn.LeakyReLU(inplace=True),  # 激活函數 ReLU
            torch.nn.Linear(1024, 1 * 28 * 28),  # 降維回去，輸入維度為1024，輸出維度為1*28*28
            torch.nn.Tanh()  # GAN 生成器通常使用 Tanh 作為輸出層的激活函數，將輸出值映射到 -1~1 之間
        )

    def forward(self, x): # x 為輸入的隨機噪聲，形狀為 (batch_size, input_dim) 此為 Sequential、Linear、ReLU 皆規定的輸入格式，batch_size 表示生成多少張影像，input_dim 表示生成器的輸入維。print: torch.Size([64, 100])，64 為 batch_size、100 為 input_dim
        output = self.model(x)  # 前向傳播計算輸出，output 為 (batch_size, 1*28*28) 的張量，表示每個樣本展平後的生成影像。print: torch.Size([64, 784])
        image = output.reshape(x.shape[0], 1, 28, 28) # 將生成的平坦（flat）數據轉換為影像格式(batch_size, 1, 28, 28)，便於後續處理或視覺化。x.shape[0] 為 batch_size(自己定義的)。print: torch.Size([64, 1, 28, 28])

        return image

# # 以下註解測試使用生成器生成影像
# # 設定輸入噪聲的維度
# input_dim = 100  # MNIST數據集的影像不大，不用那麼多細節，設太大會跑很久且容易過擬合
#
# # 初始化生成器
# generator = Generator(input_dim=input_dim)
#
# # 定義批量大小
# batch_size = 64
#
# # 生成隨機噪聲，為2*2的張量，大小為 (batch_size, input_dim)
# noise = torch.randn(batch_size, input_dim)  # 生成 batch_size 個樣本的隨機噪聲(初始影像)。每個樣本有 input_dim 個數值，平均值為0，標準差為1
#
# # 生成器生成影像
# fake_images = generator(noise) # 會調用生成器的 forward 方法，將 noise 作為輸入(此為 python 的 __call__ 方法)
# fake_images = fake_images.detach()  # 停止計算梯度，使之後的反向傳播不改變生成器的參數，因此，反向傳播時只有判別器的權重會進行更新，而不會改變生成器的權重。
#
# # 檢查輸出形狀
# print(fake_images.shape)  # (batch_size, 1, 28, 28)
#
# # 生成影像可視化
# import matplotlib.pyplot as plt
# # 從生成影像中取出第一張影像
# sample_image = fake_images[0].squeeze()  # squeeze() 去掉維度為1的維度，即將形狀為 (1, 28, 28) 的影像轉換為 (28, 28)
# plt.imshow(sample_image, cmap='gray')
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 準備判別器
class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(1 * 28 * 28, 1024),  # 全連接層，輸入維度為1*28*28，輸出維度為1024
            torch.nn.LeakyReLU(inplace=True),  # 激活函數 ReLU
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(p=0.3),  # 丟棄率為 30%
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(p=0.3),  # 丟棄率為 30%
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(128, 1),  # 輸出層，輸出維度為1，表示判斷輸入影像為真實影像的概率
            torch.nn.Sigmoid()  # GAN 判別器通常使用 Sigmoid 作為輸出層的激活函數(邏輯回歸那個)，將輸出值映射到 0~1 之間
        )

    def forward(self, image):  # image 為輸入的影像，形狀為 (batch_size, 1, 28, 28)。print: torch.Size([64, 1, 28, 28])，64 為 batch_size
        platImage = image.reshape(image.shape[0], -1)  # 將影像展平，形狀變為 (batch_size, 1*28*28)。print: torch.Size([64, 784])
        prob = self.model(platImage) # print: torch.Size([64, 1])
        return prob

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 準備數據集、數據加載器、優化器、損失函數
dataset = torchvision.datasets.MNIST(root='./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize(28),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.5,), (0.5,))  # 上行的 ToTensor() 將影像轉為 Tensor 並將像素值映射到 0~1 之間，這裡再將其映射到 -1~1 之間，便於後續使用 Tanh
                                     ])
                                     )

# print(len(dataset))
# for i in range(3):
#     print(dataset[i][0].shape)  # 打印第i個樣本的影像形狀([1, 28, 28]) 1表示通道數，28*28表示影像大小
#     print(dataset[i][1])  # 打印第i個樣本的標籤
#     print(dataset[i][0])  # 打印第i個樣本的影像數據

# 使用 DataLoader 加載數據集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 生成器、以及其優化器
generator = Generator(input_dim=100)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999)) # betas=(0.5, 0.999) 使其前期的學習率較大，後期的學習率較小

# 判別器、以及其優化器
discriminator = Discriminator()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))

# 損失函數，這裡使用二元交叉熵損失函數
criterion = torch.nn.BCELoss()

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 清空之前生成的影像
import os
import shutil

if os.path.exists('fake_images'):
    shutil.rmtree('fake_images')
os.makedirs('fake_images')

# 開始訓練 GAN!
for epoch in range(100):  # 訓練 100 輪
    for i, mini_batch in enumerate(dataloader):  # 每次遍歷會得到 dataloader 中一個 batch(或叫mini_batch) 的數據
        # 撈真實影像
        real_images, _ = mini_batch  # 影像包含image和label，這裡不需要label，所以用 _ 表示拋棄。print: torch.Size([64, 1, 28, 28])，64 為 batch_size

        # 生成器生成假影像，可看上面【測試使用生成器生成影像】那邊
        batch_size = real_images.shape[0]
        input_dim = 100
        noise = torch.randn(batch_size, input_dim)
        fake_images = generator(noise)

        # 訓練生成器(目標: 讓判別器判斷生成的影像為真)
        prob = discriminator(fake_images) # 判斷器認為生成的影像為真的概率
        g_loss = criterion(prob, torch.ones_like(prob)) # 計算生成器的損失，希望生成的影像被判斷為真，因此標籤為1。torch.ones_like(prob) 生成一個和 prob 形狀相同且元素全為1的張量，即將假影像的標籤設為1，或者也可以寫 torch.ones(batch_size, 1)
        g_optimizer.zero_grad()  # 梯度清零
        g_loss.backward() # 反向傳播計算梯度，由於 g_loss 是由生成器進行損失計算的，因此只會計算到生成器的參數，不會影響判別器的參數
        g_optimizer.step() # 更新生成器的權重

        # 訓練判別器(目標: 讓判別器判斷真實影像為真，生成的影像為假)
        prob_real = discriminator(real_images)
        # d_loss_real = criterion(prob_real, torch.ones_like(prob_real)) # 希望判別器將真實影像判斷為真，因此標籤為1
        d_loss_real = criterion(prob_real, torch.ones_like(prob_real) * 0.9)  # 這裡將標籤設為0.9，是為了增加一些噪音，使判別器不會過於自信
        prob_fake = discriminator(fake_images.detach()) # 注意這裡加上 detach()，因為我們只希望更新判別器的權重，不希望更新生成器的權重
        d_loss_fake = criterion(prob_fake, torch.zeros_like(prob_fake)) # 希望判別器將生成的影像判斷為假，因此標籤為0
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 每 100 個 batch 輸出一次結果
        if i % 100 == 0:
            print(f'Epoch: {epoch}, Step: {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # 每 500 個 batch 保存一次生成的影像，保存到當前目錄下的 fake_images 文件夾，每次保存 1 張
        if i % 500 == 0:
            torchvision.utils.save_image(fake_images, 'fake_images/{}_{}.png'.format(epoch, i), normalize=True)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# 最終保存生成器的模型參數和判別器的模型參數
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# 如果想斷點續訓，可以保存優化器的狀態
# # 假設 optimizer 是用來優化生成器和判別器的優化器
# torch.save({
#     'epoch': epoch,
#     'generator_state_dict': generator.state_dict(),
#     'discriminator_state_dict': discriminator.state_dict(),
#     'optimizer_generator_state_dict': optimizer_generator.state_dict(),
#     'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
# }, 'checkpoint.pth')



