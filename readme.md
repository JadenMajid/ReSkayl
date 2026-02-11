# Reskale

Implementation project to implement the GAN-upscaler defined at <a href="https://arxiv.org/pdf/1609.04802">here</a>

Model weights can be found at `model/srgan.pth`.

CLI Tool coming soon

## Todo
- [X] Create default model structure as a `pytorch.nn.module`
- [ ] Create smaller model for lower end systems
- [X] Train model with Flickr2K dataset
- [ ] Create results showcase/blogpost
- [ ] UI/UX
  - [ ] Make standalone binary CLI tool
    - [ ] User specified model as arg
    - [ ] Single file support
    - [ ] Support for all files in a directory
  - [ ] Make GUI tool(web or native? hmm)
