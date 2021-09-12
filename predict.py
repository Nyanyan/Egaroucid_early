from tensorflow.keras.models import load_model
import numpy as np

data = [
    [-0.4448905545133057, -0.6161570518354514, -0.8505714703078375, -0.9060085657829324, -0.8567248920387254, -0.815383229907226, -0.5709289265465051, -0.40482092544272075, -0.582993434576979, 1.8342647605196962, -0.9944604336414931, -0.9874217774881211, -0.969071781629158, -0.8986021660715416, -0.48969643157627435, -0.47280708230789653, -0.7882510970665121, 1.0662456240792684, 0.9999016086986658, 0.9980965045147009, 0.9985072817513045, 1.0404719925183568, -0.81749815799308, -0.6540974030938154, -0.8081048582801357, -0.9654165438947969, -0.9988467465934624, 1.0010113081121157, 0.9992399550035992, 1.0375865574558167, -0.8688394293752107, 1.4711090320810287, -0.7802332029036673, 1.0815001204709047, -0.9836478517240679, 0.9978644005554539, 0.9989182275430032, -0.9434105680851164, -0.8242300986067332, -0.6668087979270109, -0.7415114184682008, -0.8623886161583492, 1.062479801728498, -0.9580756960917343, -0.918765433597683, -0.886181313782825, -0.7728478896029016, -0.6310439560540092, -0.494296401984985, 2.1295842269229808, 1.3284677756915755, 1.216231993000693, -0.7958928028279357, -0.7394263601458194, -0.44038161114250257, -0.4501171511003634, 2.620641118216822, -0.44587119643679074, 1.658443886457464, -0.6458686102408555, -0.6376583733587261, -0.59865906034998, -0.43367485746035955, -0.3700223042526059, 2.2471609039775062, 1.6305296021870253, 1.1817239615764608, 1.1091416969563954, 1.172684339875695, -0.8108600571931632, -0.5675129778477773, -0.40366781601596885, 1.719062024409872, -0.5462818936802275, 1.0097529959534486, 1.0138988754824767, 1.0317490013790644, -0.896149441348393, -0.48848874613076615, -0.4731894040283995, 1.2707222822855715, -0.9373480473309125, -0.9998658324423347, -0.9980965045147008, -0.9984001062309512, -0.9601211172225412, 1.2233582778170173, 1.5264815634845743, 1.2395963864516928, 1.0316935742353799, 0.9989182275430013, -1.0010113081121155, -0.9992399550035991, -0.9626878488805939, 1.149030668133068, -0.6811380045153895, 1.2843041450913615, -0.9238429876723955, 1.0171881171632016, -0.9978644005554538, -0.9989182275430033, 1.0616788689332581, 1.212262826734357, 1.5010350955985519, 1.3495723949101608, 1.1607794232661295, -0.9388478210837393, 1.0434969967786085, 1.0897320306561964, 1.125104433471573, 1.2940392839356918, 1.5851977591610544, -0.49290899089063145, -0.4701568438425293, -0.752921901638349, -0.8201655347867084, 1.260555249413395, -0.7398460713055267, -0.440149952606702, -0.4496572138198837, -0.3818782325189707, -0.4479598578442564, -0.6053335680268411, -0.6472735129539343, -0.6387409686747025, 1.665796384859124, -0.43515876509510587, -0.3713646859601944, -0.5351458807139755, 0.8895826625166924, 1.164265259770568, 0.8868817434785576, 0.42062452797951966, -0.30192395391328697, 1.2372162988196977, 1.1899297599494787, -0.9451439454766208]
]

data = np.array(data)
model = load_model('param/model.h5')
predictions = model.predict(data).flatten()
print(predictions)