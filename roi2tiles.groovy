import qupath.lib.regions.RegionRequest
import qupath.lib.gui.scripting.QPEx

server = getCurrentImageData().getServer()
pixelfactor = server.getPixelCalibration().getPixelHeightMicrons()
tile_px = 512
tile_mic = tile_px * pixelfactor  

selectAnnotations();

runPlugin('qupath.lib.algorithms.TilerPlugin', '{"tileSizeMicrons": '+tile_mic+',  "trimToROI": false,  "makeAnnotations": true,  "removeParentAnnotation": false}');

def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()

def filename = server.getMetadata().getName() 

def path = buildFilePath(PROJECT_BASE_DIR, 'annotation results/', filename)
mkdirs(path)


i = 1

getAnnotationObjects().forEach(it -> {if (it.getPathClass() == getPathClass("Tumor")) {x =  it.getChildObjects().size()}})

print x

z = 1000000.intdiv(x)

print z

if (x > 1000) {

    for (annotation in getAnnotationObjects()) {
    
        roi = annotation.getROI()
    
        def request = RegionRequest.createInstance(imageData.getServerPath(),
            1, roi)
    
        x = annotation.getROI().getCentroidX()
        y = annotation.getROI().getCentroidY()
    
        String tiletype = annotation.getParent().getPathClass()
    
        if (tiletype.equals("Tumor")) {
    
            String tilename = String.format("%s_%s%s" +" x "+x+" y "+y+" " + ".tif", filename, tiletype,
            annotation.getName())
            
            a = Math.abs(new Random().nextInt() % 1000 + 1)
    
            if (a<z) {
        
                writeImageRegion(server, request, path + "/" + tilename);
                print("wrote " + tilename)
        
    }
        
            i++
        
        }
    }
}

else {

    for (annotation in getAnnotationObjects()) {
    
        roi = annotation.getROI()
    
        def request = RegionRequest.createInstance(imageData.getServerPath(),
        1, roi)
    
        x = annotation.getROI().getCentroidX()
        y = annotation.getROI().getCentroidY()
    
        String tiletype = annotation.getParent().getPathClass()
    
        if (tiletype.equals("Tumor")) {
    
            String tilename = String.format("%s_%s%s" +" x "+x+" y "+y+" " + ".tif", filename, tiletype,
            annotation.getName())
        
        
            writeImageRegion(server, request, path + "/" + tilename);
            
            print("wrote " + tilename)
                
        }
        
            i++
        
        }
}