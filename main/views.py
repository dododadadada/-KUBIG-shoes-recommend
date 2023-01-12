from django.shortcuts import render, redirect, get_object_or_404
from .models import Musinsa_w, Musinsa_m, Profile
import pandas as pd
# Create your views here.
from django.http import HttpResponse


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
    feature = get_vector('C:/shoes/mysite/' + str(img))

    #유사한 신발 찾기
    #1. 남성 무신사
    sim_m = []
    check = []
    for i in range(len(features_m_resnet)):
        if i in noshoes_m:
            pass
        else:
            sim = cos_pytorch(feature, features_m_resnet[i])
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
            sim = cos_pytorch(feature, features_w_resnet[i])
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


    man = Musinsa_m.objects.all()
    woman = Musinsa_w.objects

    context = {'upload_img': img, 'shoes_recommend': shoes_recommend,
               'codi_recommend': codi_recommend, 'similarity': best_similarity,
               'index': best_index, 'man_data': man[1], 'woman_data': woman}

    return render(request, 'main/profile.html', context)




# 이미지 특징 추출 딥러닝 코드
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# Load the pretrained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()


def get_vector(image):
    img = Image.open(image)
    img = img.convert('RGB')
    # 2. Create a PyTorch Variable with the transformed musinsa_image
    # Image transforms
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros([1, 512, 1, 512])

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed musinsa_image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

def cos_pytorch(A, B):
  cos= nn.CosineSimilarity(dim=1, eps=1e-6)
  cos_sim = cos(A, B)
  return cos_sim.numpy()[0][0][0]

#무신사 데이터 셋의 특징벡터 추출
#남성
noshoes_m= [16, 26, 39,55,69,244,245,246,285,307,309,310,311,312,314,315,316,317,318,319,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,353,354,355,356,357,358,359,360,362,363,364,365,366,414,418,421]
features_m_resnet = []
for i in range(427):
    # 신발 이미지 아닌거 빼기
  if i in noshoes_m:
    features_m_resnet.append(None)
  else:
    fea = get_vector(f"C:/shoes/mysite/static/musinsa_image/man_shoes/man_{i}.jpg")
    features_m_resnet.append(fea)
#여성
noshoes_w = [47, 77, 134, 139, 148, 149, 157, 171, 220, 273, 313, 314, 315, 319, 321, 329, 342, 352, 353, 356, 357, 397, 404, 413, 415, 424, 427, 447, 461, 477, 479, 504, 510, 513, 527, 529, 530, 537, 541, 542, 544, 545, 547, 548, 549, 550, 551, 553, 554, 555, 557, 558, 559, 560, 561, 563, 565, 566, 569, 570, 573, 574, 575, 576, 578, 579, 582, 583, 584, 585, 587, 588, 589, 590, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 620, 621, 622, 623, 624, 625, 626, 628, 629, 631, 632, 637, 638, 639, 644, 645, 647, 648, 649, 651, 652, 653, 654, 658, 659]
features_w_resnet = []
for i in range(660):
#신발 이미지 아닌거 빼기
  if i in noshoes_w:
    features_w_resnet.append(None)
  else:
    fea = get_vector(f"C:/shoes/mysite/static/musinsa_image/woman_shoes/woman_{i}.jpg")
    features_w_resnet.append(fea)