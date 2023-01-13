from django.shortcuts import render, redirect, get_object_or_404
from .models import Musinsa_w, Musinsa_m, Profile
import pandas as pd
# Create your views here.
from django.http import HttpResponse
import numpy as np
from numpy import dot
from numpy.linalg import norm


def index(request):
    return render(request, 'main/home.html')

def manselect(request):
    return render(request, 'main/man_home.html')

def womanselect(request):
    return render(request, 'main/woman_home.html')

def downloadManMusinsa(request):
    filename = 'C:/Korea university/쿠빅/22-2 추천시스템 프로젝트/all_man_recommend.csv'
    df = pd.read_csv(filename, encoding="UTF-8", na_values='nan')
    count = 0
    for i in range(len(df)):

        Musinsa_m.objects.create(prd_title = df['prd_title'][i],
        category = df['category'][i],
        brand = df['brand'][i],
        price = df['price'][i],
        codi_url = df['link'][i],
        shoes_url = df['shoes_link'][i],
        codi_rec = df['codi_to_codi_top5'][i],
        shoes_rec = df['category_to_category'][i])

    return HttpResponse(f'''
    <html>
    <body>
        <h1>카테고리 이름</h1>
        <h1>Download Complete</h1>
        {len(df)}
        {count}
        <h2>Data Types</h2>
        {df.dtypes}
    </body>
    </html>
    ''')

def downloadWomanMusinsa(request):
    filename = 'C:/Korea university/쿠빅/22-2 추천시스템 프로젝트/all_woman_recommend.csv'
    df = pd.read_csv(filename, encoding="UTF-8", na_values='nan')
    count = 0
    for i in range(len(df)):

        Musinsa_w.objects.create(prd_title = df['prd_title'][i],
        category = df['category'][i],
        brand = df['brand'][i],
        price = df['price'][i],
        codi_url = df['link'][i],
        shoes_url = df['shoes_link'][i],
        codi_rec = df['codi_to_codi_top5'][i],
        shoes_rec = df['category_to_category'][i])

    return HttpResponse(f'''
    <html>
    <body>
        <h1>카테고리 이름</h1>
        <h1>Download Complete</h1>
        {len(df)}
        {count}
        <h2>Data Types</h2>
        {df.dtypes}
    </body>
    </html>
    ''')

man = Musinsa_m.objects

def upload(request):
    return render(request,'main/upload.html')

def upload_create(request):
    #업로드한 이미지 가져옴
    form=Profile()
    try:
        form.image=request.FILES['image']
    except: #이미지가 없어도 그냥 지나가도록-!
        pass
    form.save()
    img = form.image.url
    #특징벡터 추출
    img1 = Image.open('.' + str(img))
    feature = fe.extract(img1)
    #유사한 신발 찾기
    #1. 남성 무신사
    sim_m = []
    check = []
    for i in range(len(features_m_resnet)):
        if i in noshoes_m:
            pass
        else:
            sim = cos(feature, features_m_resnet[i])
            # 중복이미지 제거
            if sim in check:
                pass
            else:
                check.append(sim)
                sim_m.append((sim, i))
    a = sorted(sim_m, reverse=True)[:5]
    recommend = []
    similarity = []
    index = []
    for j in range(5):
        recommend.append('/static/musinsa_image/man_shoes/man_' +str(a[j][1]) + '.jpg')
        recommend.append('/static/musinsa_image/man_codi/' + str(a[j][1]) + '.jpg')
        index.append([a[j][1], 'man'])
        similarity.append((a[j][0], j))
    #2. 여성 무신사
    sim_w = []
    check1 = []
    for i in range(len(features_w_resnet)):
        if i in noshoes_w:
            pass
        else:
            sim = cos(feature, features_w_resnet[i])
            if sim in check1:
                pass
            else:
                check1.append(sim)
                sim_w.append((sim, i))
    b = sorted(sim_w, reverse=True)[:5]
    for j in range(5):
        recommend.append('/static/musinsa_image/woman_shoes/woman_' + str(b[j][1]) + '.jpg')
        recommend.append('/static/musinsa_image/woman_codi/' + str(b[j][1]) + '.jpg')
        index.append([b[j][1] , 'woman'])
        similarity.append((b[j][0], j+5))
    # 유사도 큰 순으로 뽑기
    best = sorted(similarity, reverse=True)[:5]
    shoes_recommend = []
    codi_recommend = []
    best_similarity = []
    sex = []
    best_index = []
    for k in range(5):
        shoes_recommend.append(recommend[2*int(best[k][1])])
        codi_recommend.append(recommend[2 * int(best[k][1])+1])
        best_similarity.append(best[k][0])
        sex.append(index[int(best[k][1])][1])
        best_index.append(index[int(best[k][1])][0])



    context = {'upload_img': img, 'shoes_recommend': shoes_recommend,
               'codi_recommend': codi_recommend, 'similarity': best_similarity,
               'index': best_index}

    return render(request, 'main/profile.html', context)




# 이미지 특징 추출 딥러닝 코드

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image

class FeatureExtractor:
  def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

  def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = img
        x = np.expand_dims(x, axis=0) #차원추가
        x = preprocess_input(x) #모델에 필요한 형식에 이미지를 적절하게 맞추기위한 것
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

fe = FeatureExtractor()

#무신사 데이터 셋의 특징벡터 추출
#남성
noshoes_m= [16, 26, 39,55,69,244,245,246,285,307,309,310,311,312,314,315,316,317,318,319,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,353,354,355,356,357,358,359,360,362,363,364,365,366,414,418,421]
features_m_resnet = []
for i in range(427):
    # 신발 이미지 아닌거 빼기
  if i in noshoes_m:
    features_m_resnet.append(None)
  else:
    img = Image.open(f"./static/musinsa_image/man_shoes/man_{i}.jpg")
    fea = fe.extract(img)
    features_m_resnet.append(fea)
#여성
noshoes_w = [47, 77, 134, 139, 148, 149, 157, 171, 220, 273, 313, 314, 315, 319, 321, 329, 342, 352, 353, 356, 357, 397, 404, 413, 415, 424, 427, 447, 461, 477, 479, 504, 510, 513, 527, 529, 530, 537, 541, 542, 544, 545, 547, 548, 549, 550, 551, 553, 554, 555, 557, 558, 559, 560, 561, 563, 565, 566, 569, 570, 573, 574, 575, 576, 578, 579, 582, 583, 584, 585, 587, 588, 589, 590, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 620, 621, 622, 623, 624, 625, 626, 628, 629, 631, 632, 637, 638, 639, 644, 645, 647, 648, 649, 651, 652, 653, 654, 658, 659]
features_w_resnet = []
for i in range(660):
#신발 이미지 아닌거 빼기
  if i in noshoes_w:
    features_w_resnet.append(None)
  else:
    img = Image.open(f"./static/musinsa_image/woman_shoes/woman_{i}.jpg")
    fea = fe.extract(img)
    features_w_resnet.append(fea)

def cos(A, B):
  return dot(A, B)/(norm(A)*norm(B))