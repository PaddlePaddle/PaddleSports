from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="./SNMOT")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test","challenge"])