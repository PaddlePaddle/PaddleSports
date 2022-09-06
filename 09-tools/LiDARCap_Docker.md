### 欢迎使用LiDARCap

LiDARCap是首个基于激光雷达点云的远距离、无标记的三维人体动作捕捉框架。传统的动作捕捉方法通过在人体身上穿戴标记来记录动作。该技术起源于动画制作领域，随后被推广到虚拟现实、体育训练等领域。然而，基于标记的动作捕捉方法的成本高昂，采集流程复杂。虽然基于图像的无标记动作捕捉方法部分解决了该问题，但是图像无法获取精确的位置，而且由距离增加带来的图像质量退化也会导致算法性能下降。相对于图像，激光雷达可以获得远距离场景的三维信息，具有精度高、稳定性好等优点，更适合用于捕捉远距离的三维人体动作。

LiDARCap方法包括一个动态点云时空编码器、一个逆动力学求解器和一个SMPL 优化器。LiDARCap 和现有的基于图像的方法在 LiDARHuman26M 上进行了对比实验，并且在其他公开数据集上进行了泛化性评估。实验结果表明，在远距离场景下，LiDARCap 取得较高精度。


#### LiDARCap镜像的下载及配置

LiDARCap的镜像可以从以下百度网盘链接下载

链接：https://pan.baidu.com/s/1Mwka0Us8ejPy7J0_p5zx9w

提取码：9p1w


下载完镜像之后，对lidarcap-runtime.rar解压，解压得到lidarcap- runtime.tar

之后在装有docker的ubuntu (18.04或者更高) 命令行下执行以下命令

    sudo docker load <lidarcap-runtime.tar

之后查看ubuntu的镜像信息

    sudo docker images

将REPOSITORY, TAG 两个位置都是none的镜像的标签修改为正确的标签

假定该镜像的image id为 6201ee3d0944

执行

    sudo docker tag 6201ee3d0944 lidarcap-runtime:v0

之后便可以使用对应的LiDARCap的docker镜像了。


###  LiDARCap镜像的使用

可以通过在服务器端启动LiDARCap服务器，之后在客户端发送人体点云数据，进而从服务器端获取到pose参数的方式来使用LiDARCap

#### 服务器端

    sudo docker run --gpus '"device=0"' -d -p 5000:5000 lidarcap-runtime:v0 /bin/sh /run.sh

这个命令是利用GPU 0，在宿主机的端口5000，对外开放一个lidarcap服务。

如果要在宿主机使用其他端口，可以将-p 5000:5000的第一个5000修改为对应端口

#### 客户端

在客户端可以构建POST请求，将需要进行姿态估计的点云数据发送到lidarcap服务器的对应接口中，如果是在ubuntu中，假定客户端与服务器端在同一台计算机中，可以执行以下命令获取pose信息。

    curl -d @test_human_points.json -X POST http://127.0.0.1:5000/lidarcap --header "Content-Type: application/json"

其中test_human_points.json是一个包含了人体点云数据的json文件，可以在以下链接下载
链接：https://pan.baidu.com/s/1TJl76hYRaBZ38C9l0LYfNQ 
提取码：chk9 
这个json文件中，有一个key: "human_points"，value是一个 [6, 16, 512, 3]的点云数据，其中 6是 batch_size, 16是时序点云的长度，512是点云数量，3是点云数据的x,y,z维度。

服务器会返回[batch_size, T, 24, 3]的数据，其中T是时序长度，24是24个人体关节点，3是关节点的x,y,z。

#### LiDARCap论文
LiDARCap: Long-range Marker-less 3D Human Motion Capture with LiDAR Point Clouds, Li. et al. CVPR 2022

#### LiDARHuman项目
http://www.lidarhumanmotion.net/



