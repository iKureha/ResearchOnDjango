# -*- encoding: utf-8 -*-

from django.shortcuts import render

def test(request):
    context = {}
    context['hello'] = 'Hello World!'
    return render(request, 'li.html', context)
