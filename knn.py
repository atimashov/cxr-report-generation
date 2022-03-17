from bioCaption.data.downloads import DownloadData
from bioCaption.models.tagModels.knn import Knn
import json

reDownload = False
reTrain = False
resultsFolder = 'results_tag'

if(reDownload):
    downloads = DownloadData()
    #download the iu_xray dataset in the current directory
    downloads.download_iu_xray()

if(reTrain):
    knn = Knn('iu_xray/iu_xray.json', 'iu_xray/iu_xray_images/', resultsFolder)
    best_k = knn.knn()
    knn.test_knn(best_k)


generated_path = "./" + resultsFolder + "/" results_knn.json"
truth_path = "./iu_xray/iu_xray_auto_tags.json"

generated = json.load(open(generated_path))
truth = json.load(open(truth_path))

correct = 0
total = 0
for image, tags in generated.items():
    total += len(truth[image])
    correct += len(set(tags).intersection(set(truth[image])))

print(correct/total)
