#include <ctime>
#include <fstream>
#include <iostream>
#include <algorithm>



int main() {
    std::cout << "Title of the publication:\n";
    std::string title, fname, date;
    std::getline(std::cin, title);
    fname = title;
    std::transform(fname.begin(), fname.end(), fname.begin(),
            [](unsigned char c){ return std::tolower(c); });
    for (auto &t: fname) if (t == ' ') t = '-';
    fname += ".md";

    std::string path = "content/archive/" + fname;
    std::ofstream file(path);

    time_t now = time(0);
    struct tm  tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

    file << "+++\ntitle = \"" << title << "\"\ndate = " << buf << "\n+++\n\n\n";
    file.close();
    std::string call = "/usr/bin/vim \"+call cursor(6,0)\" " + path;
    system(call.c_str());
    std::cout << "File created at: " << path << std::endl;
}
