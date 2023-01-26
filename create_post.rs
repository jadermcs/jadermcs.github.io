use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("Title of the publication:");

    let mut title = String::new();
    io::stdin()
        .read_line(&mut title)
        .expect("Failed to read line.");
    title = title.trim().to_string();
    let fname = title.to_lowercase().replace(" ", "-") + ".md";

    let path = Path::new("content/archive").join(fname);
    let display = path.display();

    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why),
        Ok(file) => file,
    };

    let output = format!(
        "+++\ntitle = \"{}\"\ndate = {}\n+++\n\n\n",
        title, "2023-01-01"
    );

    match file.write_all(output.as_bytes()) {
        Err(why) => panic!("couldn't write to {}: {}", display, why),
        Ok(_) => println!("successfully wrote to {}", display),
    }
    Command::new("/usr/bin/zsh")
        .arg("-c")
        .arg(format!("vim \"+call cursor(6,0)\" {}", display))
        .spawn()
        .expect("vi command failed")
        .wait()
        .expect("editor failed");
}
