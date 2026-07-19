from flask import Flask, render_template, request, session, redirect, url_for
from geopy.distance import geodesic
import cv2
import os
from pyzbar.pyzbar import decode
