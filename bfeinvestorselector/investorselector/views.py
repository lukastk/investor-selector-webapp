from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import loader

from .investor_selector_lsi import *

from .forms import DescriptionForm

def index(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = DescriptionForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            desc = form.cleaned_data['description']
            ti = TopicModelIndexer(folder_dir="./investorselector/index_models")
            ti.index_database(desc)
            db = ti.db.sort_values("Similarity", ascending=False)

            sim_precentage = [ str(round(s*100, 2)) + "%" for s in db["Similarity"].tolist() ]
            request.session['result'] = list(zip(db["Organisation"].tolist(), db["Website"].tolist(), sim_precentage))

            return HttpResponseRedirect('/investorselector/result')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = DescriptionForm()

    template = loader.get_template('investorselector/index.html')
    context = {'form' : form}
    return HttpResponse(template.render(context, request))

def result(request):
    template = loader.get_template('investorselector/result.html')

    res = request.session.get('result')
    context = {'result' : res}

    return HttpResponse(template.render(context, request))
