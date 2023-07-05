import static qupath.lib.gui.scripting.QPEx.*

server  = getCurrentServer()
path = server.getPath()
downsample = 1.0

name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
pathOutput = buildFilePath(PROJECT_BASE_DIR, "patches", name)
pathOutput_Tumor = buildFilePath(PROJECT_BASE_DIR, "patches", name, "Tumor")
pathOutput_Negative = buildFilePath(PROJECT_BASE_DIR, "patches", name, "Negative")
mkdirs(pathOutput)
mkdirs(pathOutput_Tumor)
mkdirs(pathOutput_Negative)


i = 1
j = 1
for (annotation in getAnnotationObjects()){
    roi = annotation.getROI()
    request = RegionRequest.createInstance(path, downsample, roi)
    if (annotation.toString().contains("Tumor")){
        writeImageRegion(server, request, pathOutput_Tumor + "/Tumor_" + i + "_" + roi.toString() + '.tiff')
        i = i + 1
        }
     else{
        writeImageRegion(server, request, pathOutput_Negative + "/Negative_" + j + "_" + roi.toString() + '.tiff')
        j = j + 1
        }
     
}