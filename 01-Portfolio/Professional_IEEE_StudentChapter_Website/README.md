# IEEE VJIT Student Chapter Website

A static, responsive website presenting the IEEE VJIT Student Branch, its chapters, activities, events, team, gallery, and contact section. It is implemented with HTML, CSS, and browser-side JavaScript, with no application server or database.

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Interactions](#interactions)
- [Limitations](#limitations)

## Features

- Responsive navigation with a mobile menu
- Branch overview and IEEE SSIT / IEEE CS chapter tabs
- Expandable announcements
- Past-event and gallery category filters
- Gallery lightbox with Escape-key dismissal
- Scroll-to-top control and section navigation
- Client-side contact form feedback

## Technology Stack

| Area | Technology |
|------|------------|
| Markup | HTML5 |
| Styling | CSS3 |
| Interactivity | Vanilla JavaScript |
| Fonts | Google Fonts (Montserrat and Open Sans) |

## Project Structure

```text
Professional_IEEE_StudentChapter_Website/
├── index.html       # Main page
├── index1.html      # Additional HTML page
├── styles.css       # Layout, responsive styles, and visual design
├── script.js        # Navigation, tabs, filters, lightbox, and form behavior
└── README.md
```

## Run Locally

No package installation or build step is required. Open `index.html` in a modern browser.

For local HTTP testing, serve the project directory with any static file server and open the URL it provides. The site has no backend API.

## Interactions

The JavaScript controls the mobile menu, chapter tabs, announcement expansion, event and gallery filters, image lightbox, smooth section navigation, and scroll-to-top button. The contact form displays a confirmation alert in the browser and resets the form; it does not send or store a message.

## Limitations

- The contact form is a front-end demonstration only. Connect it to a backend or form service to receive submissions.
- The project has no automated tests, package scripts, or production build configuration.
- Content and event information are static and must be edited in the HTML.
