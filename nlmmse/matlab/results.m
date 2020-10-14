close all;clear;clc;
%experiment 1
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

fig = figure('Units','inches',...
'Position',[0 0 7 4],...
'PaperPositionMode','auto');

x_snr = [-6.58249031200274,-6.18561361846295,-5.46420286771958,-4.26941059762383,-2.51529842475687,-0.242729299331180,2.41256942348224,5.30499912218031,8.32853718806938,11.4199775780973,14.5453872217894,17.6874946158789,20.8377388206352,23.9919312724467,27.1480355888674,30.3050647633731,33.4625411181353,36.6202336393892,39.7780306434733,42.9358781459669];
y_optimal = [-0.619100624177964,-0.660455458691819,-0.738833490150500,-0.927251287995877,-1.33579854464517,-1.95002530816507,-2.91118505358461,-3.91188779331992,-5.22512355301361,-6.56658199261872,-7.71967442550900,-8.97576905130873,-9.95193764688905,-10.6637216722364,-11.2941207277486,-11.8136737272375,-11.8739490676789,-11.8036377413611,-12.2655178311187,-12.1134818119713];
y_ssfn = [-0.447975244573882,-0.499306841155385,-0.589589769909751,-0.820364830854633,-1.31672862437220,-2.14548883923237,-3.62913695826874,-5.54590634951198,-8.21595073878540,-11.0022020443288,-13.4752908328947,-15.7896286927383,-18.1592416154551,-20.0893501599424,-22.0239176099921,-23.8475428392465,-25.1635342636534,-26.8697515270431,-28.5545710845006,-30.1641197274040];
y_elm = [-0.897646891218553,-0.798645168844431,-0.887019050102097,-1.16522646198850,-1.80291539931001,-2.85327573617060,-4.82438885295286,-8.79923171955261,-14.4293792441198,-18.9060574934210,-22.0991276657209,-25.3707300726165,-28.4265092008580,-31.5909492933880,-34.7534849016691,-37.8812656842734,-41.0915363164839,-44.2771465957668,-47.3916514765553,-50.5628571796373];

% 100 epochs, 10 monte carlo, Time taken: 2764.583151001483 seconds (2000 samples)
y_cnn =[1.3691090914376247, 1.3375488463459284, 0.9778611688655692, 0.5680739276222084, -0.1145978735956725, -1.3018117699482197, -3.254148000382547, -6.406999775130155, -10.051329275790428, -13.713169836855606, -16.69703473347801, -18.89920920241855, -21.542263187004423, -23.723318800719277, -25.71379075572738, -27.6018103758545, -30.777099314170297, -31.148120336071955, -30.6742315365086, -32.779313192389665];

% 100 epochs 10 monte carlo, Time taken: 7107.008136276156 secs (2000 samples)
y_resnet = [0.2361651643235138, 0.24007228766894645, 0.0555170479473817, -0.21874604873440717, -0.6145699509674228, -1.5273614297910343, -3.2189409305921775, -6.217250257583512, -10.144382561850506, -14.198429436360623, -16.401851053281263, -16.56273971717024, -15.8499530921769, -14.91521489428656, -13.128288041802609, -9.701694218400805, -6.694970858697898, -4.581505038443992, -3.1946849177672085, -2.1712653003146256];

file_name = 'mmse_1';
folder_name = 'plots/A';
        
plot_title = {['P = 10, ' 'Q = 10']
              ['b = 50']};
                      
x_label = 'SNR (dB)';
legend_label = {'Optimal' 'SSFN' 'ELM', 'FCNN', 'ResNet'};
y_label = 'NMSE (dB)';
title_position = [6, 1];
    
xlim([-7 20])
ylim([-28 5])
hold on;grid on;
plot(x_snr, y_elm,'-.rp','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_ssfn,'-.bs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_optimal,'-.cs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_cnn,'-.ms','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_resnet,'-.gs','MarkerSize',4, 'LineWidth', 1.5)

set_plot_property(fig, x_label, y_label, legend_label, plot_title, file_name, folder_name, title_position);


% Experiment 2
clear;clc;

set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

fig = figure('Units','inches',...
'Position',[0 0 7 4],...
'PaperPositionMode','auto');

x_snr = [-9.68964805145288,-6.00543752513709,-2.32122699882130,1.36298352749449,5.04719405381028,8.73140458012607,12.4156151064419,16.0998256327576,19.7840361590734,23.4682466853892,27.1524572117050,30.8366677380208,34.5208782643366,38.2050887906524,41.8892993169682,45.5735098432840,49.2577203695998,52.9419308959156,56.6261414222313,60.3103519485471];

y_ssfn = [-0.221617594223743,-0.748214056410246,-1.76912917356155,-3.32200033997224,-5.28538034153814,-7.43518882283053,-9.51262633762008,-11.4624902734896,-13.1915327506238,-14.9448225243680,-16.3152785620315,-17.8127918639152,-20.4363319430677,-21.1075944594157,-22.4530186774573,-24.7258356568917,-28.2896019681765,-29.9065716110384,-31.1922657494257,-31.4054153832996];
y_elm = [-0.412483296188728,-0.830818416013505,-1.60790226186896,-2.67447124196227,-3.88928988307842,-5.18705175852238,-6.45568979351439,-7.46770753994237,-8.03783358099588,-8.57644399243239,-8.65428660862239,-8.72309942659548,-8.83822324230320,-8.88841946509406,-8.88830167302199,-8.73030439729074,-8.85520086003406,-8.82917501812493,-8.96283895445919,-8.80076136695869];
%y_optimal = [-0.470101074174025,-1.08175516839384,-2.30183584795708,-4.34904080316527,-7.56664604674090,-10.5710214319322,-12.1939513279714,-14.3265356371579,-15.6443897635673,-17.3137350759865,-18.7539367427971,-20.4472929913871,-22.3177586908587,-23.4298206341838,-25.5611370408005,-27.0757537948937,-27.6303350397897,-30.5354443152142,-30.9091791548110,-35.4681102776391];
y_optimal = [-0.526953981536623,-1.15472637600188,-2.28651468670781,-4.33369679564193,-7.38037763276032,-10.5950082411336,-12.5956614208518,-14.1244058082424,-15.7472154864211,-17.1956415823451,-18.8212116741785,-20.2908125925863,-22.0583497003047,-23.1451362281665,-25.0543721107595,-26.8634188838309,-28.6997424606484,-30.1791646668366,-32.6311087531502,-33.5668408268020];

% 100 epochs 10 monte carlo, Time taken: 2236.3013101739343 secs (3000 samples)
y_cnn = [1.7333145372850556, 0.9931814418174814, -0.5451839070361244, -2.9927511639905124, -6.162138322342432, -8.604731069451661, -10.39820932239479, -11.951603959205839, -13.793540570844192, -14.980926962560625, -17.39316094898148, -17.922859109611565, -19.78018868809443, -18.356667419122108, -19.621925856259356, -22.24122619129757, -20.27542342180574, -19.40939639103056, -18.77038257169206, -19.270114002476294];

% 100 epochs 10 monte carlo, Time taken: 7829.29932128638 secs (3000 samples)
y_resnet = [0.38772830135095643, -0.02977917536438793, -1.1123429136524994, -3.062884151168468, -5.865737101190396, -8.030160821509845, -9.799086210840128, -11.174637130388316, -12.899042177477531, -13.854756224889078, -15.438769765205638, -15.664268822290236, -16.70223565101898, -16.18339505442429, -16.844417193678677, -17.405386010684147, -16.61732296704505, -16.53585070434643, -16.080364152635994, -16.238576123255488];

file_name = 'mmse_2';
folder_name = 'plots/A';
        
plot_title = {['P = 10, ' 'Q = 10']
              ['a = 10']};
                      
x_label = 'SNR (dB)';
legend_label = {'Optimal' 'SSFN' 'ELM', 'FCNN', 'ResNet'};
y_label = 'NMSE (dB)';
title_position = [5, 1];
    
xlim([-10 20])
ylim([-25 5])
hold on;grid on;
plot(x_snr, y_optimal,'-.rp','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_ssfn,'-.bs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_elm,'-.cs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_cnn,'-.ms','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_snr, y_resnet,'-.gs','MarkerSize',4, 'LineWidth', 1.5)

set_plot_property(fig, x_label, y_label, legend_label, plot_title, file_name, folder_name, title_position);


% Experiment 3
clear;clc;
   

set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

fig = figure('Units','inches',...
'Position',[0 0 7 4],...
'PaperPositionMode','auto');

x_P = [5,10,15,20,25,30,35,40,45,50,55,60];

y_elm = [-3.77706243427540,-7.11831281652270,-8.55826136223141,-9.13954033963565,-9.20946566164523,-9.61160262689245,-9.72483848237665,-9.70543730153350,-9.82769694998121,-9.76728537990634,-9.84092243284003,-10.0193702118889];
y_ssfn = [-3.98412180532627,-10.7184457967332,-15.7027962676417,-18.3685089438994,-19.8897718948438,-21.2958202836801,-22.1334334037457,-22.9652308163399,-23.6357905995224,-24.0916190942298,-24.7288456219708,-25.1488549292355];
y_optimal = [-9.69682456696468,-15.9809731793238,-18.9141677106463,-21.0820301478103,-22.5334906114339,-23.7601132262488,-24.5390357030825,-25.4176392231621,-25.9150239730957,-26.4083432755858,-27.0085716972532,-27.4835386669620];

% 100 epochs 10 monte carlo, Time taken: 967.2929353471845 - secs (3000 samples)
y_cnn = [-7.63983404981221, -13.500247103109665, -17.409314165738145, -19.368486963550797, -20.807580842441126, -21.812180624253358, -22.541777491200005, -23.150143674687907, -23.291758725775964, -23.767384457087868, -24.00748768503652, -24.42178504335296];
% 100 epochs 10 monte carlo, Time taken: - 4789.718890514225 secs (3000 samples)
y_resnet = [-7.372033179480695, -12.61049692591882, -15.31020754077552, -15.963111048717018, -16.226867828464, -16.412006148750542, -16.255817988771277, -16.19120905262658, -16.084626152547884, -15.708326261373326, -15.198415201691331, -15.154797839868408];

file_name = 'mmse_3';
folder_name = 'plots/A';
        
plot_title = {['SNR = 20.388' ', Q = 10']
                ['a = 10' ' and b = 1']};
                      
x_label = 'Dimension of observation (P) w.r.t. a given Dimension of data (Q=10)';
legend_label = {'Optimal' 'SSFN' 'ELM', 'FCNN', 'ResNet'};
%legend_label = {'Optimal' 'SSFN' 'ELM'};
y_label = 'NMSE (dB)';
title_position = [20, 1];
    
xlim([5 35])
ylim([-25 5])

hold on;grid on;
plot(x_P, y_optimal,'-.rp','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_P, y_ssfn,'-.bs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_P, y_elm,'-.cs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_P, y_cnn,'-.ms','MarkerSize',4, 'LineWidth', 1.5) 
hold on;grid on;
plot(x_P, y_resnet,'-.gs','MarkerSize',4, 'LineWidth', 1.5)

set_plot_property(fig, x_label, y_label, legend_label, plot_title, file_name, folder_name, title_position);


% Experiment 4
clear;clc;
   

set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

fig = figure('Units','inches',...
'Position',[0 0 7 4],...
'PaperPositionMode','auto');

x_P = [100,600,1100,1600,2100,2600,3100,3600,4100,4600,5100];

y_elm = [-4.59876183338631,-5.76986754053788,-6.18639075632537,-6.24292921655185,-6.11993468773858,-6.35318695695191,-6.18496379112307,-6.27608617931827,-6.29542557446687,-6.33376822223397,-6.20701469443507];
y_ssfn = [-4.15132745910133,-7.62819407542815,-8.42566006405274,-8.52876201137729,-8.49455597151111,-8.75128761960647,-8.61966883214828,-8.76192863950255,-8.81887890136541,-8.90711158815412,-8.67790840610204];
y_optimal = [-9.94025016284133,-9.91371632523645,-9.88744265426026,-9.86975051743694,-10.3024136929392,-10.1665965238157,-10.2668086072184,-10.1959355041857,-9.96526202665526,-9.97405268613073,-10.0857644472729];

% 100 epochs 10 monte carlo, Time taken: - 889.06 secs
y_cnn = [-4.694537601932385, -7.697608916692877, -8.077838547688156, -8.123871422802473, -8.542226633502413, -8.856523945330503, -8.702621191139466, -8.872166254355575, -9.215213908987124, -8.920900879167236, -8.6787029026192];
% 100 epochs 10 monte carlo, Time taken: - 4116.627269800752 secs
y_resnet = [-0.30189217764556453, -3.2222280328967363, -5.39550996062192, -6.786605247552984, -7.515129727234779, -8.076534630090967, -7.806935811419583, -7.975081698246543, -8.406636490561322, -7.895779768353411, -7.817799977705031];

file_name = 'mmse_4';
folder_name = 'plots/A';
        
plot_title = {['SNR = 15.3995' ', P = 10, Q = 10']
                ['a = 5' ' and b = 1']};
            
                      
x_label = 'Size of dataset';
legend_label = {'Optimal' 'SSFN' 'ELM', 'FCNN', 'ResNet'};
%legend_label = {'Optimal' 'SSFN' 'ELM'};
y_label = 'NMSE (dB)';
title_position = [2500, 1];
    
xlim([0 5200])
ylim([-25 5])

hold on;grid on;
plot(x_P, y_optimal,'-.rp','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_P, y_ssfn,'-.bs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_P, y_elm,'-.cs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_P, y_cnn,'-.ms','MarkerSize',4, 'LineWidth', 1.5) 
hold on;grid on;
plot(x_P, y_resnet,'-.gs','MarkerSize',4, 'LineWidth', 1.5)

set_plot_property(fig, x_label, y_label, legend_label, plot_title, file_name, folder_name, title_position);


% Experiment 5
clear;clc;
   

set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

fig = figure('Units','inches',...
'Position',[0 0 7 4],...
'PaperPositionMode','auto');

x_SNR = [-9.66951194796886,-5.98530142165308,-2.30109089533729,1.38311963097850,5.06733015729429,8.75154068361008,12.4357512099259,16.1199617362417,19.8041722625575,23.4883827888732,27.1725933151890,30.8568038415048,34.5410143678206,38.2252248941364,41.9094354204522,45.5936459467680,49.2778564730838,52.9620669993996,56.6462775257154,60.3304880520311];

y_elm = [12.3987783998535,8.80911404708405,5.34879932337876,1.91596673939144,-1.19226703599834,-3.86958959990290,-5.94912205222344,-7.12745379745184,-7.93160988317050,-8.24556643244930,-8.37202260144473,-8.46596100518626,-8.48990812198878,-8.46334820707446,-8.30041458302958,-8.42496679036410,-8.44007208577285,-8.52532912888043,-8.52301986130993,-8.5287634852894];
y_ssfn = [11.8197385804801,8.38900922020865,4.99701079484935,1.66472009136908,-1.34875039192799,-4.53733035303421,-7.77904292795787,-10.6309881590582,-13.0173252858689,-14.5020153492911,-15.4575163408211,-15.7435602689699,-16.2541956497912,-16.0725203187876,-15.5179551406996,-16.0526992195396,-16.1201187128725,-16.1688300414301,-16.2459135535851,-16.1097317271398];
y_optimal = [-0.421714093927250,-1.14478525811248,-2.11095224573500,-4.32985376771928,-7.65403166046689,-10.5359723172566,-12.8610028882363,-14.4087792593162,-15.8456392907752,-17.3427060487740,-18.8143152564564,-20.4899459453561,-22.2609940435066,-23.8836756024488,-25.2886991420008,-27.6627794495277,-28.7540936820882,-30.9443706406519,-31.9401541654660,-34.7924284232875];

% 100 epochs 10 monte carlo, Time taken: - secs (3000 samples)
y_cnn = [-7.63983404981221, -13.500247103109665, -17.409314165738145, -19.368486963550797, -20.807580842441126, -21.812180624253358, -22.541777491200005, -23.150143674687907, -23.291758725775964, -23.767384457087868, -24.00748768503652, -24.42178504335296];
% 100 epochs 10 monte carlo, Time taken: - secs (3000 samples)
y_resnet = [-1.5101852228073842, -1.827965628152689, -1.58226347092838, -1.2814338302395898, -1.245461435053603, -0.9352503329940773, -0.8526634451649839, -0.5122800783355155, -0.8414266695203041, -0.6082904903626614, -0.4549114168887348, -0.3392400963612924];

file_name = 'mmse_5';
folder_name = 'plots/A';
        
plot_title = {['P = 10, ' 'Q = 10']
                ['a = 10']};
            
                      
x_label = 'SNR (dB)';
legend_label = {'Optimal' 'SSFN' 'ELM', 'FCNN', 'ResNet'};
%legend_label = {'Optimal' 'SSFN' 'ELM'};
y_label = 'NMSE (dB)';
title_position = [12, 10];
    
xlim([-10 35])
ylim([-25 15])

hold on;grid on;
plot(x_SNR, y_optimal,'-.rp','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_SNR, y_ssfn,'-.bs','MarkerSize',4, 'LineWidth', 1.5)
hold on;grid on;
plot(x_SNR, y_elm,'-.cs','MarkerSize',4, 'LineWidth', 1.5)
% hold on;grid on;
% plot(x_P, y_cnn,'-.ms','MarkerSize',4, 'LineWidth', 1.5) 
% hold on;grid on;
% plot(x_P, y_resnet,'-.gs','MarkerSize',4, 'LineWidth', 1.5)

set_plot_property(fig, x_label, y_label, legend_label, plot_title, file_name, folder_name, title_position);
