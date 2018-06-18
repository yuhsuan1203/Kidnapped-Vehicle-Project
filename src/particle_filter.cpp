/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	//cout << particles.size() << endl;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]); 
	normal_distribution<double> dist_theta(theta, std[2]);

	for(int i = 0; i < num_particles; i++)
	{
		Particle initial_particle;
		initial_particle.id = i;// + 1;
		initial_particle.x = dist_x(gen);
		initial_particle.y = dist_y(gen);
		initial_particle.theta = dist_theta(gen);
		initial_particle.weight = 1.0;
		particles.push_back(initial_particle);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// x_f = x0 + v / yaw_rate * [sin(theta0 + yaw_rate * dt) - sin(theta0)]
	// y_f = y0 + v / yaw_rate * [cos(theta0) - cos(theta0 + yaw_rate * dt)]
	// theta_f = theta0 + yaw_rate * dt
	// cout << "prediction" << endl;
	normal_distribution<double> dx(0, std_pos[0]);
	normal_distribution<double> dy(0, std_pos[1]);
	normal_distribution<double> dtheta(0, std_pos[2]);

	for(int i = 0; i < num_particles; i++)
	{
		if(fabs(yaw_rate) > 0.00001)
		{
			particles[i].x = particles[i].x + velocity / yaw_rate * ( sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta) );
			particles[i].y = particles[i].y + velocity / yaw_rate * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t) );
			particles[i].theta = particles[i].theta + yaw_rate * delta_t;
		}
		else
		{
			particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			particles[i].theta = particles[i].theta;
		}

		particles[i].x += dx(gen);
		particles[i].y += dy(gen);
		particles[i].theta += dtheta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i = 0; i < observations.size(); i++)
	{
		int id = -1;
		double min_distance = numeric_limits<double>::max(); 
		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			double current_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if(current_distance < min_distance)
			{
				id = predicted[j].id;
				min_distance = current_distance;
			}
		}
		observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i < num_particles; i++)
	{
		vector<LandmarkObs> predicted;
		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if(distance < sensor_range)
			{
				LandmarkObs LmOb;
				LmOb.id = map_landmarks.landmark_list[j].id_i;
				LmOb.x = map_landmarks.landmark_list[j].x_f;
				LmOb.y = map_landmarks.landmark_list[j].y_f;
				predicted.push_back(LmOb);
			}
		}

		vector<LandmarkObs> observations_transformed;
		for(unsigned int k = 0; k < observations.size(); k++)
		{
			LandmarkObs LmOb_transformed;
			LmOb_transformed.id = observations[k].id;
			LmOb_transformed.x = particles[i].x + cos(particles[i].theta) * observations[k].x - sin(particles[i].theta) * observations[k].y;
			LmOb_transformed.y = particles[i].y + sin(particles[i].theta) * observations[k].x + cos(particles[i].theta) * observations[k].y;
			observations_transformed.push_back(LmOb_transformed);
		}

		dataAssociation(predicted, observations_transformed);

		particles[i].weight = 1.0;

		for(unsigned int l = 0; l < observations_transformed.size(); l++)
		{
			double mean_x, mean_y;
			for(unsigned int m = 0; m < predicted.size(); m++)
			{
				if(observations_transformed[l].id == predicted[m].id)
				{
					mean_x = predicted[m].x;
					mean_y = predicted[m].y;
					break;
				}
			}

			double dx, dy;
			dx = observations_transformed[l].x - mean_x;
			dy = observations_transformed[l].y - mean_y;
			double prob = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]) * exp(-((dx * dx / (2 * std_landmark[0] * std_landmark[0])) + (dy * dy / (2 * std_landmark[1] * std_landmark[1]))));
			particles[i].weight *= prob;
		}
	}
	
	double sum = 0.0;
	for(int i = 0; i < num_particles; i++)
	{
		sum += particles[i].weight;
	}

	for(int i = 0; i < num_particles; i++)
	{
		particles[i].weight /= sum;
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> particles_update;
	discrete_distribution<>d(weights.begin(), weights.end());

	for(int i = 0; i < num_particles; i++)
	{
		int index = d(gen);
		particles_update.push_back(particles[index]);
	}

	particles.clear();
	particles.assign(particles_update.begin(), particles_update.end());
	weights.clear();
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
