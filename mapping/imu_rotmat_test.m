imu_mat = imu_table_07{:,:};
time = imu_mat(:,1);
time = time - time(1);
imu_yaw = imu_mat(:,11);
imu_pitch = imu_mat(:,12);
imu_roll = imu_mat(:,13);
imu_yaw = imu_yaw;
imu_pitch = imu_pitch;
imu_roll = imu_roll;
imu_acc_x = imu_mat(:,2);
imu_acc_y = imu_mat(:,3);
imu_acc_z = imu_mat(:,4) - 9.81;

% Source: https://robotics.stackexchange.com/questions/11178/kalman-filter-gps-imu/11180#11180
width = 1;
length = 2;
height = 0.5;
shapeTop = [...
    -width/4, -length/2; ...
    -width/4, length/4; ...
    width/4, length/4; ...
    width/4, -length/2];
shapeTop(:,3) = ones(4,1)*height/2;
shapeBottom = [...
    -width/2, -length/2; ...
    -width/2, length/2; ...
    width/2, length/2; ...
    width/2, -length/2];
shapeBottom(:,3) = ones(4,1)*height/2;
shapeBottom(:,3) = shapeBottom(:,3) - ones(4,1)*height;
shapeVertices = [shapeTop;shapeBottom];
shapeFaces = [...
    1,2,3,4,1; ...
    5,6,7,8,5; ...
    8,5,1,4,8; ...
    7,6,2,3,7; ...
    5,6,2,1,5; ...
    8,7,3,4,8];
shapeColor = [0.6 0.6 1]; % Light blue

figure(1);
subplot(4,1,1);
shapePatch = patch(...
    'Faces', shapeFaces, ...
    'Vertices', shapeVertices, ...
    'FaceColor', shapeColor);
axis equal;
xlabel('X-Axis')
ylabel('Y-Axis')
zlabel('Z-Axis')
view([60,20]);

imu_acc_transformed = zeros(size(imu_acc_x, 1), 3);
for i = 1:size(imu_acc_x, 1)
    rotMatrix = eul2rotm([imu_yaw(currentSample), imu_pitch(currentSample), imu_roll(currentSample)], 'ZYX')
    imu_acc_transformed(i,:) = (rotMatrix * [imu_acc_x(i); imu_acc_y(i); imu_acc_z(i)]).';
end
subplot(4,1,2);
%plot(imu_acc_transformed(:,1));
%plot(imu_acc_x);
plot(imu_roll);
subplot(4,1,3);
%plot(imu_acc_transformed(:,2));
%plot(imu_acc_y);
plot(imu_pitch);
subplot(4,1,4);
%plot(imu_acc_transformed(:,3));
%plot(imu_acc_z);
plot(imu_yaw);


tic
while true
    elapsedTime = toc*4;
    if elapsedTime > time(end)
        break;
    end
    currentSample = find(time>=elapsedTime,1)
    rotMatrix = eul2rotm([imu_yaw(currentSample), imu_pitch(currentSample), imu_roll(currentSample)], 'ZYX')
    tempVertices = shapeVertices;
    tempVertices = (rotMatrix*tempVertices.').';
    set(shapePatch,'Vertices',tempVertices);
    drawnow;
    pause(0.1);
end


