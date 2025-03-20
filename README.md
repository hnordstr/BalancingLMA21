These are the file for tuning and running PIM, where LMA21ImbalanceModel is the main model.

The following scripts must be used to run the main model:

- LMA21ImbalanceModel
- Gurobi_Model
- RESshares
- WindModel
- SolarModel
- DemandModel

To run the model, correct data paths must be customized for reading and storing data in the LMA21ImbalanceModel.

Also, correct cmatrix paths must be defined within the WindModel, SolarModel and DemandModel.

Input data to run the model with current tuning is available for download at: https://drive.google.com/drive/folders/10peTlcp4eJS4qt1NlCvPjF0DS0ird4AJ?usp=drive_link

For tuning the model for another setup, the user is referred to the following papers: https://ieeexplore.ieee.org/document/10202966 and https://www.sciencedirect.com/science/article/pii/S0960148124003422?via%3Dihub

The scripts use for tuning is available in this repository. However, these should rather be seen as a guide for setting up the tuning procedure. The tuning then requires some manual work as well.
