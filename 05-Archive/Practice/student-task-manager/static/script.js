async function loadTasks() {
    const res = await fetch("/api/tasks");
    const data = await res.json();
    console.log(data);
}

window.onload = loadTasks;
