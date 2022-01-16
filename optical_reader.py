import cv2
import numpy as np
import pandas as pd

"""
kağıdın üzerindeki cevaplar dikdörtgen içine alınmalıdır
dikdörtgenin köşeleri belirlenip warp perspective yapılır
paper.jpg'deki gibi işlenmeye hazır hale getirilir
"""

img = cv2.imread("paper.jpg")
img2 = img.copy()

img2 = cv2.blur(img2,(2,2))
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img2 = cv2.threshold(img2,140,255,cv2.THRESH_BINARY)
img2 = img2[1]
img2 = img2[::,40::]

"""
her bir soru ayrı satır resimlerine bölmek için 
resimin yüksekliği bölmek istediğimiz sayıya tam bölünebilmeli
her bir satırı da A B C D E gibi şıklara bölmek için
genişliğin de şık sayısına tam bölünebilmesi gerek
"""

height1 = img2.shape[0]
#574 % 9 --> 7 (yukarıdan 7 satır pixel silebiliriz)
img2 = img2[7::,:]
height2 = img2.shape[0]
#print(height1," --> ",height2)  #574 --> 567

width1 = img2.shape[1]
#373 % 5 --> 3 (kenarlardan 3 sütun pixel silebiliriz)
img2 = img2[::,:370]
width2 = img2.shape[1]
#print(width1," --> ",width2)    #373 --> 370

questions = np.vsplit(img2,9)               # 9 soru
#selections = np.hsplit(questions[0],5)     # 5 şık

verilen_cevaplar = []

for q in questions:
    selections = np.hsplit(q,5)
    question_list = []
    for index,select in enumerate(selections):
        resim_orj = cv2.resize(select,(400,400))
        resim = cv2.blur(resim_orj,(3,3))
        resim = cv2.Canny(resim,100,100)
        all_points = []
        contours,_ = cv2.findContours(resim,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            length = cv2.arcLength(contour,True)
            cornerPoints = cv2.approxPolyDP(contour,0.02*length,True)
            x,y,w,h = cv2.boundingRect(cornerPoints)
            all_points.append((x,y,x+w,y+h))
        x_locs = []
        y_locs = []
        c_locs = []
        d_locs = []
        for i in all_points:
            x_locs.append(i[0])
            y_locs.append(i[1])
            c_locs.append(i[2])
            d_locs.append(i[3])
        X = min(x_locs)
        Y = min(y_locs)
        C = max(c_locs)
        D = max(d_locs)

        parca = resim_orj[Y:D,X:C]
        density = 0
        count = 0
        for i in parca:
            for j in i:
                density = density + j
                count = count + 1
        density = density / count
        
        #düşük yoğunluk siyah pixellerin fazla olduğunu
        #işaretlenme eşiğiyle birlikte tahmin eder
        if density < 100:  
            question_list.append(1)
        else:
            question_list.append(0)
    verilen_cevaplar.append(question_list)

answersTrue_str = ["E","C","D","B","C","D","A","B","C"]
#gerçek cevap anahtarı

answersTrue = []
for i in answersTrue_str:
    if i == "A":
        answersTrue.append(0)
    if i == "B":
        answersTrue.append(1)
    if i == "C":
        answersTrue.append(2)
    if i == "D":
        answersTrue.append(3)
    if i == "E":
        answersTrue.append(4)

cevaplar = []
for i in verilen_cevaplar:
    if max(i) == 0:
        cevaplar.append(-1)
    else:
        index = i.index(max(i))
        cevaplar.append(index)

cevaplar_str = []
for i in cevaplar:
    if i == -1:
        cevaplar_str.append("-")
    if i == 0:
        cevaplar_str.append("A")
    if i == 1:
        cevaplar_str.append("B")
    if i == 2:
        cevaplar_str.append("C")
    if i == 3:
        cevaplar_str.append("D")
    if i == 4:
        cevaplar_str.append("E")

dataFrame = pd.DataFrame()
dataFrame["Doğru Cevap"] = pd.Series(answersTrue_str)
dataFrame["İşaretlenen"] = pd.Series(cevaplar_str)

soru_sonuc = []
dogru_sayisi = 0
for i in range(len(dataFrame.index)):
    if dataFrame["Doğru Cevap"][i] == dataFrame["İşaretlenen"][i]:
        soru_sonuc.append("Doğru")
        dogru_sayisi = dogru_sayisi + 1
    else:
        soru_sonuc.append("Yanlış")

dataFrame["Doğruluk"] = pd.Series(soru_sonuc)

yanlis_sayisi = len(dataFrame.index) - dogru_sayisi
result_text1 = "{} sorudan; {} doğru {} yanlış cevaplanmıştır.".format(len(dataFrame.index),dogru_sayisi,yanlis_sayisi)
result_text2 = "Net puan %{}".format(str(int((100*dogru_sayisi)/len(dataFrame.index))))

print(dataFrame)
print()
print(result_text1)
print(result_text2)