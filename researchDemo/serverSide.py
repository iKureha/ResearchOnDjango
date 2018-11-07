# -*- encoding: utf-8 -*-
import sys
sys.path.append("/Users/wang 1/researchDemo/researchDemo/Research/")

from django.shortcuts import render
from django.views.decorators import csrf
# import stop_words 
from researchTest import *

# get user_name
def compute(request):
    return_to_main = {}
    if request.POST:
        return_to_main['num'] = number_of_following(request.POST["user_name"])

        #####################
        # for real, get data from twitter
        return_to_main['lda'] = runLDA(request.POST["user_name"])

        #####################
        # for show, using local content
        #timeline = local_test()

        #return_to_main["lda"] = lda_clustering(timeline)
        return_to_main['cos'] = runCos(request.POST["user_name"], return_to_main['lda'])
        return_to_main['flag'] = 1
    return render(request, "main.html", return_to_main)
