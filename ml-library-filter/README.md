# ML filter for library browsing

This is a short and unfinished demo of a tool that allows to filter your music collection by training a small ML model using songs in your music library.

## How to use

Open `Library ML model filter demo.ipynb`. Execute all the cells.

In the widget on the bottom, select a folder containing drum'n'bass music to analyze.

Click "Load directory". Enable the "Analyze new songs" checkmark and then click the orange "Load song annotations" button. This will load the songs and annotate the songs in the music with a descriptor of the music timbre.

After the annotation is complete, you can go to the second tab. You'll see a list of your music. Then you can choose to select songs and add them to the "selection" or the "rejection". Songs in the selection are songs that you want to find more similar songs to in your own collection. Songs in the rejection are songs that are not similar (partially or at all) to the songs you want to find.  
After you have added some songs to the selection, click "Train Model" to train the logistic regression classifier. This classifier will then classify all other songs in the collection by whether they could belong to the selection or rejection, based on the songs that are already in it. You can confirm or reject both suggestions and unsuggestions, and then retrain the model. This should refine the ML-based filter for your own library, so that you hopefully can get suggestions that you are looking for :) 

Note that this demo is a WIP, and the efficacy of the model, of the filtering process or of the features is still untested.

