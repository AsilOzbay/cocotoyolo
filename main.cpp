#include <iostream>
#include <fstream>
#include <string>
#include <vector>


using namespace std;

void jpgxy(string filepath,double & width, double & height){


    FILE *image;
    int size, i = 0;
    unsigned char *data;

    image =fopen(filepath.c_str(),"rb");  // open JPEG image file
    if(!image){
        printf("Unable to open image \n");
    }
    fseek(image,  0,  SEEK_END);
    size = ftell(image);
    fseek(image,  0,  SEEK_SET);
    data = (unsigned char *)malloc(size);
    fread(data, 1, size, image);
/* verify valid JPEG header */
    if(data[i] == 0xFF && data[i + 1] == 0xD8 && data[i + 2] == 0xFF && data[i + 3] == 0xE0) {
        i += 4;
        /* Check for null terminated JFIF */
        if(data[i + 2] == 'J' && data[i + 3] == 'F' && data[i + 4] == 'I' && data[i + 5] == 'F' && data[i + 6] == 0x00) {
            while(i < size) {
                i++;
                if(data[i] == 0xFF){
                    if(data[i+1] == 0xC0) {
                        height = data[i + 5]*256 + data[i + 6];
                        width = data[i + 7]*256 + data[i + 8];
                        break;
                    }
                }
            }
        }
    }
    fclose(image);
}

int main()
{
    //reading file
    string row,maininput;

// Read from the text file
    ifstream MyReadFile("C:\\Users\\sikeratar\\CLionProjects\\odgttoyolo\\train.txt");

// Use a while loop together with the getline() function to read the file line by line
    while (getline (MyReadFile, row)) {
        // Output the text from the file
        maininput.append(row);
    }
// Close the file
    MyReadFile.close();
    //converting to char array
    int len=maininput.length();
    char* maininputchararray = new char[len ];
    for (int i = 0; i <len; i++) {
        maininputchararray[i] = maininput[i];
    }

    maininput.clear();
    string id="",output="";
    double headxy[4],bodyxy[4];
    double imgheight,imgwight;
    bool mask=false;
    bool hbox= true,fbox= false,firstiteration= true;
    for (int i = 0; i < len; ++i) {

            if (maininputchararray[i]=='m'&&maininputchararray[i+1]=='a'&&maininputchararray[i+2]=='s'&&maininputchararray[i+3]=='k'){
                mask=true;
            }

            if (maininputchararray[i]=='I'&&maininputchararray[i+1]=='D'){
                mask=false;
                if (!firstiteration){
                    ofstream MyFile("C:\\Users\\sikeratar\\CLionProjects\\odgttoyolo\\dataset\\labels\\"+id+".txt");
                    MyFile << output;
                    MyFile.close();
                    id="";output="";
                }
               for (int j = i+6; ; ++j) {
                  if (maininputchararray[j]=='"'){
                      i=j;
                      firstiteration= false;

                      string img="C:\\Users\\sikeratar\\CLionProjects\\odgttoyolo\\dataset\\images\\"+id+".jpg";
                      jpgxy(img,imgwight,imgheight);


                      break;
                   }
               id+=maininputchararray[j];
            }
        }

            else if(!mask&&hbox&&maininputchararray[i]=='h'&&maininputchararray[i+1]=='b'&&maininputchararray[i+2]=='o'&&maininputchararray[i+3]=='x')   {
                string tempdigitconvernter;
                double xy[4];
                bool xycounter[3];
                xycounter[0]= true;
                for (int j = i+8; ; ++j) {
                    if (maininputchararray[j]=='"'){
                        xy[3]= stoi(tempdigitconvernter);

                        hbox=false;
                        fbox= true;
                        i=j;
                        headxy[0]=(xy[0] + xy[2] / 2) / imgwight;
                        headxy[1]=(xy[1] + xy[3] / 2) / imgheight;
                        headxy[2]=xy[2] / imgwight;
                        headxy[3]=xy[3] / imgheight;
                        for (int k = 0; k < 4; ++k) {
                            if(headxy[k]>1){headxy[k]=1;}
                            if(headxy[k]<0){headxy[k]=0;}
                        }
                        output+="0 "+to_string(headxy[0])+" "+to_string(headxy[1])+" "+to_string(headxy[2])+" "+to_string(headxy[3])+"\n";
                        break;
                    }
                    if (maininputchararray[j]==','){
                        for (int k = 0; k < 3; ++k) {
                            if(xycounter[k]== true){
                                xy[k]= stoi(tempdigitconvernter);
                                tempdigitconvernter="";
                                xycounter[k]= false;
                                xycounter[k+1]= true;
                                break;
                            }
                        }
                    } else{
                        tempdigitconvernter+=maininputchararray[j];
                    }

                }
            }

            else if(!mask&&fbox&&maininputchararray[i]=='v'&&maininputchararray[i+1]=='b'&&maininputchararray[i+2]=='o'&&maininputchararray[i+3]=='x')   {
                string tempdigitconvernter="";
                double xy[4];
                bool xycounter[3];
                xycounter[0]= true;
                xycounter[1]= false;
                xycounter[2]= false;
                xy[0]=0;
                xy[1]=0;
                xy[2]=0;
                xy[3]=0;
                for (int j = i+8; ; ++j) {
                    if (maininputchararray[j]=='"'){
                        xy[3]= stoi(tempdigitconvernter);
                        hbox= true;
                        fbox= false;
                        i=j;
                        bodyxy[0]=(xy[0] + xy[2] / 2) / imgwight;
                        bodyxy[1]=(xy[1] + xy[3] / 2) / imgheight;
                        bodyxy[2]=xy[2] / imgwight;
                        bodyxy[3]=xy[3] / imgheight;
                        for (int k = 0; k < 4; ++k) {
                            if(bodyxy[k]>1){bodyxy[k]=1;}
                            if(bodyxy[k]<0){bodyxy[k]=0;}
                        }
                        output+="1 "+to_string(bodyxy[0])+" "+to_string(bodyxy[1])+" "+to_string(bodyxy[2])+" "+to_string(bodyxy[3])+"\n";
                        break;
                    }
                    if (maininputchararray[j]==','){
                        for (int k = 0; k < 3; ++k) {
                            if(xycounter[k]== true){
                                xy[k]= stoi(tempdigitconvernter);
                                tempdigitconvernter="";
                                xycounter[k]= false;
                                xycounter[k+1]= true;
                                break;
                            }
                        }
                    } else{
                        tempdigitconvernter+=maininputchararray[j];
                    }
                }
            }
        }

    //spit

    return 0;
}