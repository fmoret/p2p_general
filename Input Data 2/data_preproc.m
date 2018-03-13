Village1=[7 9 30 39 64];    	% Postcode 2259
Village2=[5 61 69 70 127];   	% Postcode 2261
Village3=[59 87 88 110 293];	% Postcode 2290
Set_village=[Village1,Village2,Village3];

PV = Australian_data_1_hour.PV(:,Set_village);
Load = Australian_data_1_hour.Load(:,Set_village);
Aggregated_Load = Australian_data_1_hour.Aggregated_Load(:,Set_village);
PostCode = Australian_data_1_hour.Post_code(Set_village);

data = struct('PV', PV, 'Load', Load, 'Aggregated_Load', Aggregated_Load);