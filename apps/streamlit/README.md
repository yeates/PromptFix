```
conda create -n promptfix python=3.10 -y
conda activate promptfix


ln -s ~/projects/wgong/PromptFix/configs ./configs
ln -s ~/projects/wgong/PromptFix/checkpoints ./checkpoints
ln -s ~/projects/wgong/PromptFix/stable_diffusion ./stable_diffusion
ln -s ~/projects/wgong/PromptFix/utils ./utils

```


## Issues

### OSError: [Errno 28] inotify watch limit reached

```
cat /proc/sys/fs/inotify/max_user_watches
# 65536

# Create or edit the config file
sudo nano /etc/sysctl.conf

# Add this line to the file:
fs.inotify.max_user_watches=524288

# Save and apply changes
sudo sysctl -p
```